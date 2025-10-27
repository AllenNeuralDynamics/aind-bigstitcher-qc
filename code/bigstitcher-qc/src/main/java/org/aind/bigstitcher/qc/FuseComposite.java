package org.aind.bigstitcher.qc;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import bdv.ViewerImgLoader;
import bdv.cache.CacheControl;
import bdv.img.cache.VolatileGlobalCellCache;
import bdv.util.MipmapTransforms;
import mpicbg.spim.data.generic.sequence.BasicViewDescription;
import mpicbg.spim.data.sequence.ImgLoader;
import mpicbg.spim.data.sequence.ViewDescription;
import mpicbg.spim.data.sequence.ViewId;
import mpicbg.spim.data.sequence.ViewSetup;
import net.imglib2.Dimensions;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.ByteArray;
import net.imglib2.realtransform.AffineTransform3D;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import net.preibisch.mvrecon.Threads;
import net.preibisch.mvrecon.fiji.spimdata.SpimData2;
import net.preibisch.mvrecon.fiji.spimdata.XmlIoSpimData2;
import net.preibisch.mvrecon.fiji.spimdata.imgloaders.OMEZarrAttibutes;
import net.preibisch.mvrecon.process.downsampling.DownsampleTools;
import net.preibisch.mvrecon.process.fusion.FusionTools;
import net.preibisch.mvrecon.process.fusion.lazy.LazyFusionTools;
import net.preibisch.mvrecon.process.fusion.transformed.TransformView;
import net.preibisch.mvrecon.process.n5api.N5ApiTools;
import net.preibisch.mvrecon.process.n5api.N5ApiTools.MultiResolutionLevelInfo;
import net.preibisch.mvrecon.process.fusion.transformed.TransformVirtual;
import org.janelia.saalfeldlab.n5.Compression;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.janelia.saalfeldlab.n5.universe.StorageFormat;
import org.janelia.saalfeldlab.n5.universe.metadata.ome.ngff.v04.OmeNgffMultiScaleMetadata;
import org.slf4j.LoggerFactory;
import util.URITools;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import picocli.CommandLine.ParseResult;

public class FuseComposite {

    private static final float DEFAULT_DISPLAY_MAX = 100f;
    private static float displayMax = DEFAULT_DISPLAY_MAX;

    private static final double DEFAULT_ANISOTROPY = 1.0;
    private static final double DEFAULT_DOWNSAMPLING = 8.0;
    private static final int DEFAULT_MAX_DOWNSAMPLING_LEVELS = 7;
    private static final int DEFAULT_INTERPOLATION = 1;
    private static final String UNIT_PIXEL = "micrometer";

    private static Path omeZarrRoot;
    private static Integer userDefinedBlockSize;
    private static int maxDownsamplingLevels = DEFAULT_MAX_DOWNSAMPLING_LEVELS;
    private static int interpolation = DEFAULT_INTERPOLATION;

    private enum ColorMode {
        VIEW,
        ILLUMINATION
    }

    @Command(
            name = "CreateVolumeOverlay",
            mixinStandardHelpOptions = true,
            description = "Fuses the full downsampled volume into a multicolor ARGB overlay for QC."
    )
    private static final class CliOptions {
        @Parameters(index = "0", paramLabel = "XML", description = "Path to the BigStitcher XML dataset.")
        Path xmlPath;

        @Parameters(index = "1", paramLabel = "OUTPUT", description = "Root directory where the OME-Zarr volume will be written.")
        Path outputRoot;

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

        final ColorMode colorMode;
        try {
            colorMode = parseColorMode(options.colorMode);
        } catch (IllegalArgumentException e) {
            commandLine.getErr().println(e.getMessage());
            commandLine.usage(commandLine.getErr());
            return;
        }

        omeZarrRoot = options.outputRoot.toAbsolutePath().normalize();
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

        final List<ViewId> views = collectPresentViews(spimData);
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

        final OverlayCompositor overlay = prepareOverlayCompositor(
                spimData,
                regs,
                views,
                globalInterval,
                interpolation,
                colorMode
        );

        exportVolumeAsOmeZarr(overlay, options.datasetName);
        clearImgLoaderCache(spimData);
        System.out.println("Finished generating volume overlay.");
    }

    private static List<ViewId> collectPresentViews(final SpimData2 spimData) {
        final List<ViewId> views = new ArrayList<>();
        for (final ViewDescription vd : spimData.getSequenceDescription().getViewDescriptions().values()) {
            if (spimData.getSequenceDescription().getMissingViews() == null ||
                    !spimData.getSequenceDescription().getMissingViews().getMissingViews().contains(vd)) {
                views.add(vd);
            }
        }
        return views;
    }

    private static ColorMode parseColorMode(final String value) {
        if (value == null)
            return ColorMode.ILLUMINATION;
        final String normalized = value.trim().toLowerCase();
        switch (normalized) {
            case "view":
                return ColorMode.VIEW;
            case "illumination":
                return ColorMode.ILLUMINATION;
            default:
                throw new IllegalArgumentException("Unsupported color mode: " + value);
        }
    }

    private static Interval globalBounds(
            final Collection<? extends ViewId> views,
            final Map<ViewId, AffineTransform3D> regs,
            final Map<ViewId, ? extends BasicViewDescription<ViewSetup>> viewDescriptions
    ) {
        long[] min = null;
        long[] max = null;

        for (final ViewId vid : views) {
            final Interval bounds = boundsForView(vid, regs, viewDescriptions);
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

    private static Interval boundsForView(
            final ViewId vid,
            final Map<ViewId, AffineTransform3D> regs,
            final Map<ViewId, ? extends BasicViewDescription<?>> viewDescriptions
    ) {
        final BasicViewDescription<?> vd = viewDescriptions.get(vid);
        final AffineTransform3D model = regs.get(vid);
        if (vd == null || model == null)
            return null;

        final Dimensions size = vd.getViewSetup().getSize();
        if (size == null)
            return null;

        return Intervals.largestContainedInterval(model.estimateBounds(new FinalInterval(size)));
    }

    private static OverlayCompositor prepareOverlayCompositor(
            final SpimData2 spimData,
            final Map<ViewId, AffineTransform3D> regs,
            final Collection<? extends ViewId> allViews,
            final Interval cropBB,
            final int interpolation,
            final ColorMode colorMode
    ) {
        final Map<ViewId, ? extends BasicViewDescription<ViewSetup>> vds =
                spimData.getSequenceDescription().getViewDescriptions();

        final ArrayList<ViewId> overlapping = LazyFusionTools.overlappingViewIds(
                cropBB,
                allViews,
                regs,
                LazyFusionTools.assembleDimensions(allViews, vds),
                LazyFusionTools.defaultAffineExpansion
        );

        final List<RandomAccessibleInterval<FloatType>> transformedSources = new ArrayList<>(overlapping.size());
        final List<Integer> sourceIds = new ArrayList<>(overlapping.size());
        for (final ViewId vid : overlapping) {
            final AffineTransform3D model = regs.get(vid).copy();

            final double[] usedDownsampleFactors = new double[3];
            final RandomAccessibleInterval<FloatType> input = DownsampleTools.openDownsampled(
                    spimData.getSequenceDescription().getImgLoader(),
                    vid,
                    model,
                    usedDownsampleFactors
            );

            RandomAccessibleInterval<FloatType> transformed = TransformView.transformView(
                    input, model, cropBB, 0, interpolation);

            transformed = Views.zeroMin(transformed);
            transformedSources.add(transformed);

            final BasicViewDescription<ViewSetup> vd = vds.get(vid);
            sourceIds.add(colorMode == ColorMode.ILLUMINATION ? illuminationIdOf(vd) : vid.getViewSetupId());
        }

        final int n = transformedSources.size();
        final int[] palette;
        final int[] colorIndexForSource = new int[n];

        if (colorMode == ColorMode.ILLUMINATION) {
            final LinkedHashMap<Integer, Integer> illumColorIndex = new LinkedHashMap<>();
            for (final Integer illumId : sourceIds)
                illumColorIndex.computeIfAbsent(illumId, ignored -> illumColorIndex.size());
            palette = distinctPalette(illumColorIndex.size());
            for (int idx = 0; idx < n; idx++)
                colorIndexForSource[idx] = illumColorIndex.get(sourceIds.get(idx));
        } else {
            palette = distinctPalette(n);
            for (int idx = 0; idx < n; idx++)
                colorIndexForSource[idx] = idx;
        }

        final Interval outInterval = FusionTools.getFusedZeroMinInterval(cropBB);
        final long[] cropMin = new long[3];
        cropBB.min(cropMin);

        return new OverlayCompositor(outInterval, transformedSources, palette, colorIndexForSource, cropMin);
    }

    private static int illuminationIdOf(final BasicViewDescription<ViewSetup> vd) {
        if (vd == null || vd.getViewSetup() == null || vd.getViewSetup().getIllumination() == null)
            return -1;
        return vd.getViewSetup().getIllumination().getId();
    }

    private static void exportVolumeAsOmeZarr(
            final OverlayCompositor overlay,
            final String datasetName
    ) throws IOException {
        if (omeZarrRoot == null)
            throw new IllegalStateException("OME-Zarr output root is not configured");

        final Path outputFolder = omeZarrRoot.resolve(datasetName).normalize();
        final Path containerPath = outputFolder.getParent() == null
                ? Paths.get(outputFolder.toString() + ".ome.zarr")
                : outputFolder.getParent().resolve(outputFolder.getFileName().toString() + ".ome.zarr");

        Files.createDirectories(containerPath.getParent());
        deleteRecursivelyIfExists(containerPath);

        final long[] spatialDims = overlay.spatialDimensions();
        final long[] dims5d = new long[] { spatialDims[0], spatialDims[1], spatialDims[2], 3, 1 };
        final int[] blockSize = defaultBlockSize(spatialDims);
        final int[][] downsamplings = defaultDownsamplings(spatialDims);
        final Compression compression = new GzipCompression(5);

        final ExecutorService executor = Executors.newFixedThreadPool(Threads.numThreads());
        try {
            try (final N5Writer writer = URITools.instantiateN5Writer(
                    StorageFormat.ZARR,
                    URITools.toURI(containerPath.toString()))) {

                final MultiResolutionLevelInfo[] pyramid = N5ApiTools.setupMultiResolutionPyramid(
                        writer,
                        level -> Integer.toString(level),
                        DataType.UINT8,
                        dims5d,
                        compression,
                        blockSize,
                        downsamplings
                );

                writeS0Blocks(overlay, writer, pyramid[0], executor);
                writeDownsampledLevels(writer, pyramid, executor);
                writeOmeNgffMetadata(writer, pyramid, containerPath.getFileName().toString(), overlay.worldMin3d());
            }
        } finally {
            shutdownExecutor(executor);
        }

        System.out.println("Exported multiscale OME-Zarr volume to " + containerPath);
    }

    private static void writeS0Blocks(
            final OverlayCompositor overlay,
            final N5Writer writer,
            final MultiResolutionLevelInfo levelInfo,
            final ExecutorService executor
    ) {
        final long[] dims3d = overlay.spatialDimensions();
        final int[] blockSize3d = new int[] {
                levelInfo.blockSize[0],
                levelInfo.blockSize[1],
                levelInfo.blockSize[2]
        };

        final List<long[][]> grid = N5ApiTools.assembleJobs(dims3d, blockSize3d);
        final List<Future<?>> futures = new ArrayList<>();

        for (long[][] gridBlock : grid) {
            final long[][] blockCopy = cloneGridBlock(gridBlock);
            futures.add(executor.submit(() -> {
                final long[] blockMin = blockCopy[0];
                final long[] blockSize = blockCopy[1];
                final long[] blockMax = new long[3];
                for (int d = 0; d < 3; d++)
                    blockMax[d] = Math.min(dims3d[d] - 1, blockMin[d] + blockSize[d] - 1);

                final int sizeX = (int)(blockMax[0] - blockMin[0] + 1);
                final int sizeY = (int)(blockMax[1] - blockMin[1] + 1);
                final int sizeZ = (int)(blockMax[2] - blockMin[2] + 1);

                final ArrayImg<UnsignedByteType, ByteArray> blockImg = ArrayImgs.unsignedBytes(sizeX, sizeY, sizeZ, 3, 1);
                overlay.renderBlock(blockMin, blockImg);

                final long[] gridOffset = new long[] { blockCopy[2][0], blockCopy[2][1], blockCopy[2][2], 0, 0 };
                N5Utils.saveNonEmptyBlock(blockImg, writer, levelInfo.dataset, gridOffset, new UnsignedByteType());
                return null;
            }));
        }

        waitForFutures(futures);
    }

    private static void writeDownsampledLevels(
            final N5Writer writer,
            final MultiResolutionLevelInfo[] pyramid,
            final ExecutorService executor
    ) {
        if (pyramid.length <= 1)
            return;

        for (int level = 1; level < pyramid.length; level++) {
            final MultiResolutionLevelInfo current = pyramid[level];
            final MultiResolutionLevelInfo previous = pyramid[level - 1];

            final long[] dims3d = new long[] {
                    current.dimensions[0],
                    current.dimensions[1],
                    current.dimensions[2]
            };

            final int[] blockSize3d = new int[] {
                    current.blockSize[0],
                    current.blockSize[1],
                    current.blockSize[2]
            };

            final List<long[][]> grid = N5ApiTools.assembleJobs(dims3d, blockSize3d);
            final List<Future<?>> futures = new ArrayList<>();

            for (long[][] gridBlock : grid) {
                for (int channel = 0; channel < current.dimensions[3]; channel++) {
                    final long[][] blockCopy = cloneGridBlock(gridBlock);
                    final long channelIndex = channel;
                    futures.add(executor.submit(() -> {
                        N5ApiTools.writeDownsampledBlock5dOMEZARR(
                                writer,
                                current,
                                previous,
                                blockCopy,
                                channelIndex,
                                0L
                        );
                        return null;
                    }));
                }
            }

            waitForFutures(futures);
        }
    }

    private static void writeOmeNgffMetadata(
            final N5Writer writer,
            final MultiResolutionLevelInfo[] pyramid,
            final String datasetName,
            final double[] worldMin3d
    ) throws IOException {
        final Function<Integer, AffineTransform3D> levelToTransform =
                level -> {
                    final double[] downsampling = pyramid[level].absoluteDownsamplingDouble();
                    final AffineTransform3D transform = MipmapTransforms.getMipmapTransformDefault(downsampling);
                    final double[] levelTranslation = computeLevelTranslation(worldMin3d, downsampling);
                    for (int axis = 0; axis < 3; axis++)
                        transform.set(levelTranslation[axis], axis, 3);
                    return transform;
                };

        final double[] resolution = new double[] {1.0, 1.0, 1.0};

        final OmeNgffMultiScaleMetadata[] metadata = OMEZarrAttibutes.createOMEZarrMetadata(
                5,
                datasetName,
                resolution,
                UNIT_PIXEL,
                pyramid.length,
                level -> "/" + level,
                levelToTransform
        );

        writer.setAttribute("/", "multiscales", metadata);
    }

    private static double[] computeLevelTranslation(final double[] worldMin3d, final double[] absoluteDownsampling) {
        final double[] translation = new double[3];
        for (int axis = 0; axis < 3; axis++) {
            final double base = worldMin3d != null && axis < worldMin3d.length ? worldMin3d[axis] : 0.0;
            final double downsampling = absoluteDownsampling != null && axis < absoluteDownsampling.length
                    ? absoluteDownsampling[axis]
                    : 1.0;
            final double shift = 0.5 * Math.max(0.0, downsampling - 1.0);
            translation[axis] = base + shift;
        }
        return translation;
    }

    private static void waitForFutures(final List<Future<?>> futures) {
        for (Future<?> future : futures) {
            try {
                future.get();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Interrupted while processing OME-Zarr blocks", e);
            } catch (ExecutionException e) {
                throw new RuntimeException("Failed to process OME-Zarr blocks", e.getCause());
            }
        }
    }

    private static long[][] cloneGridBlock(final long[][] gridBlock) {
        final long[][] copy = new long[gridBlock.length][];
        for (int i = 0; i < gridBlock.length; i++)
            copy[i] = gridBlock[i].clone();
        return copy;
    }

    private static void shutdownExecutor(final ExecutorService executor) {
        executor.shutdown();
        try {
            if (!executor.awaitTermination(1, TimeUnit.MINUTES)) {
                executor.shutdownNow();
                executor.awaitTermination(30, TimeUnit.SECONDS);
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

    private static void deleteRecursivelyIfExists(final Path path) throws IOException {
        if (!Files.exists(path))
            return;

        try (Stream<Path> walk = Files.walk(path)) {
            walk.sorted(Comparator.reverseOrder()).forEach(p -> {
                try {
                    Files.deleteIfExists(p);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            });
        } catch (UncheckedIOException e) {
            throw e.getCause();
        }
    }

    private static int[] defaultBlockSize(final long[] spatialDims) {
        final int override = userDefinedBlockSize != null ? userDefinedBlockSize : -1;
        return new int[] {
                override > 0 ? override : blockSizeFor(spatialDims[0]),
                override > 0 ? override : blockSizeFor(spatialDims[1]),
                override > 0 ? override : blockSizeFor(spatialDims[2]),
                3,
                1
        };
    }

    private static int blockSizeFor(final long dim) {
        return (int)Math.max(1, Math.min(dim, 128));
    }

    private static int[][] defaultDownsamplings(final long[] spatialDims) {
        final List<int[]> levels = new ArrayList<>();
        levels.add(new int[] {1, 1, 1, 1, 1});

        int[] current = levels.get(0);
        for (int level = 1; level <= maxDownsamplingLevels; level++) {
            final int[] next = current.clone();
            boolean changed = false;

            for (int d = 0; d < 3; d++) {
                if (spatialDims[d] / next[d] >= 2) {
                    next[d] *= 2;
                    changed = true;
                }
            }

            if (!changed) {
                break;
            }

            levels.add(next);
            current = next;
        }

        return levels.toArray(new int[0][]);
    }

    private static void clearImgLoaderCache(final SpimData2 spimData) {
        final ImgLoader imgLoader =
                spimData.getSequenceDescription().getImgLoader();
        if (imgLoader instanceof ViewerImgLoader) {
            final CacheControl cacheControl = ((ViewerImgLoader) imgLoader).getCacheControl();
            if (cacheControl == null)
                return;
            if (cacheControl instanceof VolatileGlobalCellCache) {
                ((VolatileGlobalCellCache) cacheControl).clearCache();
                System.out.println("Clearing cache");
            } else {
                try {
                    cacheControl.getClass().getMethod("clearCache").invoke(cacheControl);
                    System.out.println("Clearing cache");
                } catch (Exception ignored) {
                    // ignore if cache cannot be cleared explicitly
                }
            }
        }
    }

    private static final class OverlayCompositor {
        private final List<RandomAccessibleInterval<FloatType>> sources;
        private final int[] palette;
        private final int[] colorIndexForSource;
        private final long[] spatialDims;
        private final double[] worldMin3d;

        OverlayCompositor(
                final Interval interval,
                final List<RandomAccessibleInterval<FloatType>> sources,
                final int[] palette,
                final int[] colorIndexForSource,
                final long[] cropMin
        ) {
            this.sources = new ArrayList<>(sources);
            this.palette = palette;
            this.colorIndexForSource = colorIndexForSource.clone();
            this.spatialDims = new long[3];
            interval.dimensions(this.spatialDims);
            this.worldMin3d = new double[] {
                    cropMin != null && cropMin.length > 0 ? cropMin[0] : 0.0,
                    cropMin != null && cropMin.length > 1 ? cropMin[1] : 0.0,
                    cropMin != null && cropMin.length > 2 ? cropMin[2] : 0.0
            };
        }

        long[] spatialDimensions() {
            return spatialDims.clone();
        }

        double[] worldMin3d() {
            return worldMin3d.clone();
        }

        void renderBlock(final long[] blockMin, final ArrayImg<UnsignedByteType, ByteArray> target) {
            final int sizeX = (int)target.dimension(0);
            final int sizeY = (int)target.dimension(1);
            final int sizeZ = (int)target.dimension(2);
            if (sizeX <= 0 || sizeY <= 0 || sizeZ <= 0)
                return;

            final ByteArray access = (ByteArray)target.update(null);
            final byte[] data = access.getCurrentStorageArray();
            final int sliceSize = sizeX * sizeY;
            final int volumeSize = sliceSize * sizeZ;
            final int[] channelOffsets = new int[] { 0, volumeSize, 2 * volumeSize };

            final int sourceCount = sources.size();
            final List<RandomAccessibleInterval<FloatType>> localSources = this.sources;
            final int[] paletteLocal = this.palette;
            final int[] colorIndex = this.colorIndexForSource;

            IntStream.range(0, sizeZ).parallel().forEach(localZ -> {
                final long globalZ = blockMin[2] + localZ;
                final long[] pos = new long[] { 0L, 0L, globalZ };
                final ArrayList<RandomAccess<FloatType>> ras = new ArrayList<>(sourceCount);
                for (int s = 0; s < sourceCount; s++)
                    ras.add(localSources.get(s).randomAccess());

                for (int y = 0; y < sizeY; y++) {
                    pos[1] = blockMin[1] + y;
                    final int rowOffset = (localZ * sizeY + y) * sizeX;
                    for (int x = 0; x < sizeX; x++) {
                        pos[0] = blockMin[0] + x;
                        int argb = 0;
                        for (int s = 0; s < sourceCount; s++) {
                            final RandomAccess<FloatType> ra = ras.get(s);
                            ra.setPosition(pos);
                            final float val = ra.get().getRealFloat();
                            if (val <= 0)
                                continue;
                            final int level = clampTo8bit((val / displayMax) * 255f);
                            if (level == 0)
                                continue;
                            final int srcColor = tint(paletteLocal[colorIndex[s]], level);
                            argb = maxBlend(argb, srcColor);
                        }

                        if (argb == 0)
                            continue;

                        final int baseIndex = rowOffset + x;
                        data[channelOffsets[0] + baseIndex] = (byte)((argb >> 16) & 0xff);
                        data[channelOffsets[1] + baseIndex] = (byte)((argb >> 8) & 0xff);
                        data[channelOffsets[2] + baseIndex] = (byte)(argb & 0xff);
                    }
                }
            });
        }
    }

    private static int tint(final int baseColor, final int level) {
        final int br = (baseColor >> 16) & 0xff;
        final int bg = (baseColor >> 8)  & 0xff;
        final int bb = (baseColor)       & 0xff;
        final int r = (br * level) / 255;
        final int g = (bg * level) / 255;
        final int b = (bb * level) / 255;
        return (0xff << 24) | (r << 16) | (g << 8) | b;
    }

    private static int maxBlend(final int dst, final int src) {
        final int r = Math.max((dst >> 16) & 0xff, (src >> 16) & 0xff);
        final int g = Math.max((dst >> 8)  & 0xff, (src >> 8)  & 0xff);
        final int b = Math.max(dst & 0xff,        src & 0xff);
        return (0xff << 24) | (r << 16) | (g << 8) | b;
    }

    private static int clampTo8bit(final float v) {
        return (v <= 0) ? 0 : (v >= 255f ? 255 : (int) v);
    }

    private static int[] distinctPalette(final int n) {
        final int[] base = new int[] {
                0xFF0000,
                0x00FF00,
                0x0000FF,
                0xFFFF00,
                0xFF00FF,
                0x00FFFF,
                0xFFA500,
                0x8000FF,
                0x00FF80,
                0xFF0080
        };
        final int[] out = new int[n];
        for (int i = 0; i < n; i++) out[i] = base[i % base.length];
        return out;
    }
}
