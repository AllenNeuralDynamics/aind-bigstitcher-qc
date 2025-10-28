package org.aind.bigstitcher.qc;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import mpicbg.spim.data.generic.sequence.BasicViewDescription;
import mpicbg.spim.data.sequence.ViewId;
import mpicbg.spim.data.sequence.ViewSetup;
import net.imglib2.Dimensions;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.util.Intervals;
import net.imglib2.realtransform.AffineTransform3D;
import net.preibisch.mvrecon.process.fusion.transformed.TransformVirtual;
import net.preibisch.mvrecon.fiji.spimdata.SpimData2;
import net.preibisch.mvrecon.fiji.spimdata.XmlIoSpimData2;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import picocli.CommandLine.ParseResult;

public class FuseCompositeOverlaps {

    // Display scaling defaults: map FloatType intensities to 0..255 before tinting.
    private static final float DEFAULT_DISPLAY_MAX = 100f;
    // Use mutable value so CLI can override defaults.
    private static float displayMax = DEFAULT_DISPLAY_MAX;

    // Choose interpolation: 1=linear, 0=nearest
    private static final int DEFAULT_INTERPOLATION = 1;
    private static int interpolation = DEFAULT_INTERPOLATION;

    private static final double DEFAULT_ANISOTROPY = 1.0;
    private static final double DEFAULT_DOWNSAMPLING = 8.0;
    private static final int DEFAULT_MAX_DOWNSAMPLING_LEVELS = 7;
    private static String omeZarrRoot;
    private static Integer userDefinedBlockSize;
    private static final String UNIT_PIXEL = "micrometer";
    private static int maxDownsamplingLevels = DEFAULT_MAX_DOWNSAMPLING_LEVELS;

    @Command(
            name = "CreateOverlays",
            mixinStandardHelpOptions = true,
            description = "Creates ARGB overlap crops and writes them as multiscale OME-Zarr datasets."
    )
    private static final class CliOptions {
        @Parameters(index = "0", paramLabel = "XML", description = "Path to the BigStitcher XML dataset.")
        Path xmlPath;

        @Parameters(index = "1", paramLabel = "OUTPUT", description = "Root directory or URI where OME-Zarr crops will be written.")
        String outputRoot;

        @Option(names = "--block-size", paramLabel = "INT", description = "Override cubic block size in voxels.")
        Integer blockSizeOverride;

        @Option(
                names = "--anisotropy",
                paramLabel = "FLOAT",
                defaultValue = "" + DEFAULT_ANISOTROPY,
                description = "Anisotropy factor applied before fusion (default: ${DEFAULT-VALUE})."
        )
        double anisotropy;

        @Option(
                names = "--downsampling",
                paramLabel = "FLOAT",
                defaultValue = "" + DEFAULT_DOWNSAMPLING,
                description = "Downsampling factor applied before fusion (default: ${DEFAULT-VALUE})."
        )
        double downsampling;

        @Option(
                names = "--display-max",
                paramLabel = "FLOAT",
                defaultValue = "" + DEFAULT_DISPLAY_MAX,
                description = "Intensity mapped to 255 in the ARGB overlay (default: ${DEFAULT-VALUE})."
        )
        double displayMax;

        @Option(
                names = "--max-downsampling-levels",
                paramLabel = "INT",
                defaultValue = "" + DEFAULT_MAX_DOWNSAMPLING_LEVELS,
                description = "Maximum multi-scale pyramid levels to generate (default: ${DEFAULT-VALUE})."
        )
        int maxDownsamplingLevels;

        @Option(
                names = "--interpolation",
                paramLabel = "INT",
                defaultValue = "" + DEFAULT_INTERPOLATION,
                description = "Interpolation mode: 0=nearest, 1=linear (default: ${DEFAULT-VALUE})."
        )
        int interpolation;
    }

    public static void main(String[] args) throws Exception {
        // Preflight: propagate AWS region to N5 S3 writer if provided
        Utils.configureS3RegionFromEnv();

        final CliOptions options = new CliOptions();
        final CommandLine commandLine = new CommandLine(options);

        final ParseResult parseResult;
        try {
            parseResult = commandLine.parseArgs(args);
        } catch (CommandLine.ParameterException parameterException) {
            commandLine.getErr().println(parameterException.getMessage());
            commandLine.usage(commandLine.getErr());
            return;
        }

        if (CommandLine.printHelpIfRequested(parseResult))
            return;

        if (options.blockSizeOverride != null && options.blockSizeOverride <= 0) {
            commandLine.getErr().println("Block size must be a positive integer.");
            commandLine.usage(commandLine.getErr());
            return;
        }

        if (options.displayMax <= 0) {
            commandLine.getErr().println("Display max must be greater than zero.");
            commandLine.usage(commandLine.getErr());
            return;
        }

        if (options.maxDownsamplingLevels < 0) {
            commandLine.getErr().println("Max downsampling levels must be zero or a positive integer.");
            commandLine.usage(commandLine.getErr());
            return;
        }

        if (options.interpolation < 0 || options.interpolation > 1) {
            commandLine.getErr().println("Interpolation must be 0 (nearest) or 1 (linear).");
            commandLine.usage(commandLine.getErr());
            return;
        }

        omeZarrRoot = Utils.normalizeOutputRoot(options.outputRoot);
        userDefinedBlockSize = options.blockSizeOverride;
        displayMax = (float) options.displayMax;
        maxDownsamplingLevels = options.maxDownsamplingLevels;
        interpolation = options.interpolation;

        final Logger rootLogger = (Logger) LoggerFactory.getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME);
        rootLogger.setLevel(Level.INFO);

        final Path xmlPath = options.xmlPath.toAbsolutePath().normalize();
        final String xml = xmlPath.toString();
        final SpimData2 spimData = new XmlIoSpimData2().load(xml);
        System.out.println("Loaded: " + xml);

        // Collect all present views (filter by channel/timepoint here if you want)
        final List<ViewId> views = Utils.collectPresentViews(spimData);
        if (views.isEmpty()) {
            System.err.println("No valid views found in dataset.");
            return;
        }
        System.out.println("Processing " + views.size() + " views.");

        // To not scale, use Double.NaN for both.
        final double anisotropy = options.anisotropy; // or Double.NaN
        final double downsampling = options.downsampling;  // or Double.NaN
        final Map<ViewId, AffineTransform3D> regs = TransformVirtual.adjustAllTransforms(
                views,
                spimData.getViewRegistrations().getViewRegistrations(),
                anisotropy,
                downsampling
        );

        final Map<ViewId, ? extends BasicViewDescription<ViewSetup>> viewDescriptions =
                spimData.getSequenceDescription().getViewDescriptions();
        final List<IlluminationBounds> illuminationBounds = coalesceByIllumination(views, regs, viewDescriptions);
        System.out.println("Coalesced into " + illuminationBounds.size() + " illumination tiles.");

        if (illuminationBounds.size() <= 1) {
            System.out.println("Single illumination tile detected; falling back to per-view overlap search.");
            processPerViewOverlaps(spimData, regs, views, viewDescriptions);
        } else {
            processPerIlluminationOverlaps(spimData, regs, views, illuminationBounds);
        }
    }

    private static void processPerIlluminationOverlaps(
            final SpimData2 spimData,
            final Map<ViewId, AffineTransform3D> regs,
            final List<ViewId> views,
            final List<IlluminationBounds> illuminationBounds
    ) throws IOException {
        for (int i = 0; i < illuminationBounds.size(); i++) {
            for (int j = i + 1; j < illuminationBounds.size(); j++) {
                final IlluminationBounds tile1 = illuminationBounds.get(i);
                final IlluminationBounds tile2 = illuminationBounds.get(j);

                final Interval overlap = Intervals.intersect(tile1.interval, tile2.interval);
                if (Intervals.isEmpty(overlap)) {
                    continue;
                }

                final CompositeUtils.CompositeBlockRenderer overlay = CompositeUtils.createCompositeBlockRenderer(
                        spimData,
                        regs,
                        views,
                        overlap,
                        interpolation,
                        CompositeUtils.ColorMode.ILLUMINATION,
                        displayMax
                );

                final String name = "overlay_tile" + tile1.illuminationId + "_tile" + tile2.illuminationId;
                exportCropAsOmeZarr(overlay, "split-affine/" + name);
                Utils.clearImgLoaderCache(spimData);

                System.out.println("Created ARGB overlay crop for tiles (" + tile1.illuminationId + "," + tile2.illuminationId + ") "
                        + Arrays.toString(overlay.spatialDimensions()));
            }
        }
    }

    private static void processPerViewOverlaps(
            final SpimData2 spimData,
            final Map<ViewId, AffineTransform3D> regs,
            final List<ViewId> views,
            final Map<ViewId, ? extends BasicViewDescription<ViewSetup>> viewDescriptions
    ) throws IOException {
        for (int i = 0; i < views.size(); i++) {
            for (int j = i + 1; j < views.size(); j++) {
                final ViewId v1 = views.get(i);
                final ViewId v2 = views.get(j);

                final Interval bb1 = Utils.boundsForView(v1, regs, viewDescriptions);
                final Interval bb2 = Utils.boundsForView(v2, regs, viewDescriptions);
                if (bb1 == null || bb2 == null)
                    continue;

                final Interval overlap = Intervals.intersect(bb1, bb2);
                if (Intervals.isEmpty(overlap)) {
                    continue;
                }

                final CompositeUtils.CompositeBlockRenderer overlay = CompositeUtils.createCompositeBlockRenderer(
                        spimData,
                        regs,
                        views,
                        overlap,
                        interpolation,
                        CompositeUtils.ColorMode.VIEW,
                        displayMax
                );

                final String name = "overlay_v" + v1.getViewSetupId() + "_v" + v2.getViewSetupId();
                exportCropAsOmeZarr(overlay, "affine/" + name);
                Utils.clearImgLoaderCache(spimData);

                System.out.println("Created ARGB overlay crop for views (" + v1.getViewSetupId() + "," + v2.getViewSetupId() + ") "
                        + Arrays.toString(overlay.spatialDimensions()));
            }
        }
    }

    private static List<IlluminationBounds> coalesceByIllumination(
            final Collection<? extends ViewId> views,
            final Map<ViewId, AffineTransform3D> regs,
            final Map<ViewId, ? extends BasicViewDescription<ViewSetup>> viewDescriptions
    ) {
        final Map<Integer, long[]> minPerIllumination = new HashMap<>();
        final Map<Integer, long[]> maxPerIllumination = new HashMap<>();
        final Map<Integer, List<ViewId>> membersPerIllumination = new HashMap<>();

        for (final ViewId vid : views) {
            final BasicViewDescription<ViewSetup> vd = viewDescriptions.get(vid);
            final AffineTransform3D model = regs.get(vid);
            if (vd == null || model == null)
                continue;

            final Dimensions size = vd.getViewSetup().getSize();
            if (size == null)
                continue;

            final Interval bounds = Intervals.largestContainedInterval(
                    model.estimateBounds(new FinalInterval(size)));

            final int illuminationId = Utils.illuminationIdOf(vd);

            final long[] min = new long[bounds.numDimensions()];
            final long[] max = new long[bounds.numDimensions()];
            bounds.min(min);
            bounds.max(max);

            final long[] aggMin = minPerIllumination.get(illuminationId);
            final long[] aggMax = maxPerIllumination.get(illuminationId);
            if (aggMin == null || aggMax == null) {
                minPerIllumination.put(illuminationId, min);
                maxPerIllumination.put(illuminationId, max);
            } else {
                for (int d = 0; d < min.length; d++) {
                    if (min[d] < aggMin[d])
                        aggMin[d] = min[d];
                    if (max[d] > aggMax[d])
                        aggMax[d] = max[d];
                }
            }

            membersPerIllumination.computeIfAbsent(illuminationId, k -> new ArrayList<>()).add(vid);
        }

        final List<IlluminationBounds> groups = new ArrayList<>(minPerIllumination.size());
        for (Map.Entry<Integer, long[]> entry : minPerIllumination.entrySet()) {
            final int illuminationId = entry.getKey();
            final long[] min = entry.getValue();
            final long[] max = maxPerIllumination.get(illuminationId);
            if (max == null)
                continue;
            final List<ViewId> members = membersPerIllumination.getOrDefault(illuminationId, new ArrayList<>());
            groups.add(new IlluminationBounds(
                    illuminationId,
                    new FinalInterval(min.clone(), max.clone()),
                    new ArrayList<>(members)));
        }

        groups.sort((a, b) -> Integer.compare(a.illuminationId, b.illuminationId));
        return groups;
    }

    private static final class IlluminationBounds {
        final int illuminationId;
        final FinalInterval interval;
        final List<ViewId> members;

        IlluminationBounds(final int illuminationId, final FinalInterval interval, final List<ViewId> members) {
            this.illuminationId = illuminationId;
            this.interval = interval;
            this.members = members;
        }
    }

    private static void exportCropAsOmeZarr(
            final CompositeUtils.CompositeBlockRenderer overlay,
            final String name
    ) throws IOException {
        final String containerLocation = resolveContainerLocation(name);
        final URI containerUri = CompositeUtils.exportCompositeAsOmeZarr(
                overlay,
                containerLocation,
                name,
                userDefinedBlockSize,
                maxDownsamplingLevels,
                UNIT_PIXEL
        );

        System.out.println("Exported multiscale OME-Zarr crop to " + containerUri);
    }

    private static String resolveContainerLocation(final String datasetName) {
        final String relative = datasetName.endsWith(".ome.zarr") ? datasetName : datasetName + ".ome.zarr";

        if (omeZarrRoot == null || omeZarrRoot.isEmpty()) {
            return relative;
        }

        if (omeZarrRoot.contains("://")) {
            final String separator = omeZarrRoot.endsWith("/") ? "" : "/";
            return omeZarrRoot + separator + relative;
        }

        return Paths.get(omeZarrRoot).resolve(relative).toString();
    }
}
