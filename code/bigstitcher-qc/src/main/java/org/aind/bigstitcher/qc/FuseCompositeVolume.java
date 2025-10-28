package org.aind.bigstitcher.qc;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import mpicbg.spim.data.generic.sequence.BasicViewDescription;
import mpicbg.spim.data.sequence.ViewId;
import mpicbg.spim.data.sequence.ViewSetup;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.realtransform.AffineTransform3D;
import net.imglib2.util.Intervals;
import net.preibisch.mvrecon.fiji.spimdata.SpimData2;
import net.preibisch.mvrecon.fiji.spimdata.XmlIoSpimData2;
import net.preibisch.mvrecon.process.fusion.transformed.TransformVirtual;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import picocli.CommandLine.ParseResult;

public class FuseCompositeVolume {

    private static final float DEFAULT_DISPLAY_MAX = 100f;
    private static float displayMax = DEFAULT_DISPLAY_MAX;

    private static final double DEFAULT_ANISOTROPY = 1.0;
    private static final double DEFAULT_DOWNSAMPLING = 8.0;
    private static final int DEFAULT_MAX_DOWNSAMPLING_LEVELS = 7;
    private static final int DEFAULT_INTERPOLATION = 1;
    private static final String UNIT_PIXEL = "micrometer";

    private static String omeZarrRoot;
    private static Integer userDefinedBlockSize;
    private static int maxDownsamplingLevels = DEFAULT_MAX_DOWNSAMPLING_LEVELS;
    private static int interpolation = DEFAULT_INTERPOLATION;

    @Command(
            name = "CreateVolumeOverlay",
            mixinStandardHelpOptions = true,
            description = "Fuses the full downsampled volume into a multicolor ARGB overlay for QC."
    )
    private static final class CliOptions {
        @Parameters(index = "0", paramLabel = "XML", description = "Path to the BigStitcher XML dataset.")
        Path xmlPath;

        @Parameters(index = "1", paramLabel = "OUTPUT", description = "Root directory or URI where the OME-Zarr volume will be written.")
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

        @Option(
                names = "--dataset-name",
                paramLabel = "NAME",
                defaultValue = "volume-overlay",
                description = "Name prefix for the OME-Zarr dataset (default: ${DEFAULT-VALUE})."
        )
        String datasetName;

        @Option(
                names = "--color-mode",
                paramLabel = "MODE",
                defaultValue = "illumination",
                description = "Color coding scheme: view or illumination (default: ${DEFAULT-VALUE})."
        )
        String colorMode;
    }

    public static void main(final String[] args) throws Exception {
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

        final CompositeUtils.ColorMode colorMode;
        try {
            colorMode = CompositeUtils.parseColorMode(options.colorMode);
        } catch (IllegalArgumentException e) {
            commandLine.getErr().println(e.getMessage());
            commandLine.usage(commandLine.getErr());
            return;
        }

        omeZarrRoot = Utils.normalizeOutputRoot(options.outputRoot);
        System.out.println("OME-Zarr output root: " + omeZarrRoot);
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

        final List<ViewId> views = Utils.collectPresentViews(spimData);
        if (views.isEmpty()) {
            System.err.println("No valid views found in dataset.");
            return;
        }
        System.out.println("Processing " + views.size() + " views.");

        final double anisotropy = options.anisotropy;
        final double downsampling = options.downsampling;
        final Map<ViewId, AffineTransform3D> regs = TransformVirtual.adjustAllTransforms(
                views,
                spimData.getViewRegistrations().getViewRegistrations(),
                anisotropy,
                downsampling
        );

        final Map<ViewId, ? extends BasicViewDescription<ViewSetup>> viewDescriptions =
                spimData.getSequenceDescription().getViewDescriptions();

        final Interval globalInterval = globalBounds(views, regs, viewDescriptions);
        if (globalInterval == null || Intervals.isEmpty(globalInterval)) {
            System.err.println("Unable to determine global bounds for the dataset.");
            return;
        }
        final long[] globalMin = new long[globalInterval.numDimensions()];
        final long[] globalMax = new long[globalInterval.numDimensions()];
        globalInterval.min(globalMin);
        globalInterval.max(globalMax);
        System.out.println("Global bounding box: " + Arrays.toString(globalMin)
                + " -> " + Arrays.toString(globalMax));

        final CompositeUtils.CompositeBlockRenderer overlay = CompositeUtils.createCompositeBlockRenderer(
                spimData,
                regs,
                views,
                globalInterval,
                interpolation,
                colorMode,
                displayMax
        );

        exportVolumeAsOmeZarr(overlay, options.datasetName);
        Utils.clearImgLoaderCache(spimData);
        System.out.println("Finished generating volume overlay.");
    }

    private static Interval globalBounds(
            final Collection<? extends ViewId> views,
            final Map<ViewId, AffineTransform3D> regs,
            final Map<ViewId, ? extends BasicViewDescription<ViewSetup>> viewDescriptions
    ) {
        long[] min = null;
        long[] max = null;

        for (final ViewId vid : views) {
            final Interval bounds = Utils.boundsForView(vid, regs, viewDescriptions);
            if (bounds == null || Intervals.isEmpty(bounds))
                continue;

            final long[] viewMin = new long[bounds.numDimensions()];
            final long[] viewMax = new long[bounds.numDimensions()];
            bounds.min(viewMin);
            bounds.max(viewMax);

            if (min == null) {
                min = viewMin;
                max = viewMax;
            } else {
                for (int d = 0; d < viewMin.length; d++) {
                    if (viewMin[d] < min[d])
                        min[d] = viewMin[d];
                    if (viewMax[d] > max[d])
                        max[d] = viewMax[d];
                }
            }
        }

        if (min == null || max == null)
            return null;
        return new FinalInterval(min, max);
    }

    private static void exportVolumeAsOmeZarr(
            final CompositeUtils.CompositeBlockRenderer overlay,
            final String datasetName
    ) throws IOException {
        final URI containerUri = CompositeUtils.exportCompositeAsOmeZarr(
                overlay,
                omeZarrRoot,
                datasetName,
                userDefinedBlockSize,
                maxDownsamplingLevels,
                UNIT_PIXEL
        );

        System.out.println("Exported multiscale OME-Zarr volume to " + containerUri);
    }

}
