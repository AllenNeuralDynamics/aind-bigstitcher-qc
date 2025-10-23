package net.preibisch.mvrecon.process.fusion.lazy;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
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
import mpicbg.spim.data.generic.sequence.BasicViewDescription;
import mpicbg.spim.data.sequence.ImgLoader;
import mpicbg.spim.data.sequence.ViewDescription;
import mpicbg.spim.data.sequence.ViewId;
import mpicbg.spim.data.sequence.ViewSetup;
import net.imglib2.*;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.ByteArray;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import net.imglib2.realtransform.AffineTransform3D;
import net.preibisch.mvrecon.process.fusion.transformed.TransformVirtual;
import net.preibisch.mvrecon.fiji.spimdata.SpimData2;
import net.preibisch.mvrecon.fiji.spimdata.XmlIoSpimData2;
import net.preibisch.mvrecon.process.downsampling.DownsampleTools;
import net.preibisch.mvrecon.process.fusion.FusionTools;
import net.preibisch.mvrecon.process.fusion.transformed.TransformView;
import net.preibisch.mvrecon.Threads;
import org.slf4j.LoggerFactory;
import bdv.util.MipmapTransforms;
import org.janelia.saalfeldlab.n5.Compression;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.janelia.saalfeldlab.n5.universe.StorageFormat;
import org.janelia.saalfeldlab.n5.universe.metadata.ome.ngff.v04.OmeNgffMultiScaleMetadata;
import net.preibisch.mvrecon.process.n5api.N5ApiTools;
import net.preibisch.mvrecon.process.n5api.N5ApiTools.MultiResolutionLevelInfo;
import net.preibisch.mvrecon.fiji.spimdata.imgloaders.OMEZarrAttibutes;
import util.URITools;

public class TestColorOverlayOverlaps {

    // Display scaling: map FloatType intensities to 0..255 before tinting.
    static final float DISPLAY_MAX = 100f;

    // Choose interpolation: 1=linear, 0=nearest
    static final int INTERPOLATION = 1;

    private static final Path OME_ZARR_ROOT = Paths.get("/results", "ome-zarr_4x");
    private static final String UNIT_PIXEL = "micrometer";
    private static final int MAX_DOWNSAMPLING_LEVELS = 5;

    public static void main(String[] args) throws Exception {
        final Logger rootLogger = (Logger) LoggerFactory.getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME);
        rootLogger.setLevel(Level.INFO);

        final String xml = "/root/capsule/data/bigstitcher_affine.split.xml";
        final SpimData2 spimData = new XmlIoSpimData2().load(xml);
        System.out.println("Loaded: " + xml);

        // Collect all present views (filter by channel/timepoint here if you want)
        final List<ViewId> views = new ArrayList<>();
        for (final ViewDescription vd : spimData.getSequenceDescription().getViewDescriptions().values()) {
            if (spimData.getSequenceDescription().getMissingViews() == null ||
                !spimData.getSequenceDescription().getMissingViews().getMissingViews().contains(vd)) {
                views.add(vd);
            }
        }
        System.out.println("Processing " + views.size() + " views.");

        // FIXME: is this correct??.
        // To not scale, use Double.NaN for both.
        final double anisotropy = 1.337; // or Double.NaN
        final double downsampling = 4;  // or Double.NaN
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

                final OverlayCompositor overlay = prepareOverlayCompositor(
                        spimData,
                        regs,
                        views,
                        overlap,
                        INTERPOLATION,
                        true
                );

                final String name = "overlay_tile" + tile1.illuminationId + "_tile" + tile2.illuminationId;
                exportCropAsOmeZarr(overlay, "split-affine/" + name);
                clearImgLoaderCache(spimData);

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

                final Interval bb1 = boundsForView(v1, regs, viewDescriptions);
                final Interval bb2 = boundsForView(v2, regs, viewDescriptions);
                if (bb1 == null || bb2 == null)
                    continue;

                final Interval overlap = Intervals.intersect(bb1, bb2);
                if (Intervals.isEmpty(overlap)) {
                    continue;
                }

                final OverlayCompositor overlay = prepareOverlayCompositor(
                        spimData,
                        regs,
                        views,
                        overlap,
                        INTERPOLATION,
                        false
                );

                final String name = "overlay_v" + v1.getViewSetupId() + "_v" + v2.getViewSetupId();
                exportCropAsOmeZarr(overlay, "affine/" + name);
                clearImgLoaderCache(spimData);

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

            final int illuminationId = illuminationIdOf(vd);

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

    private static int illuminationIdOf(final BasicViewDescription<ViewSetup> vd) {
        if (vd == null || vd.getViewSetup() == null || vd.getViewSetup().getIllumination() == null)
            return -1;
        return vd.getViewSetup().getIllumination().getId();
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

    /** Prepare metadata and transformed sources so crops can be rendered block-wise on demand. */
    static OverlayCompositor prepareOverlayCompositor(
            final SpimData2 spimData,
            final Map<ViewId, AffineTransform3D> regs,
            final Collection<? extends ViewId> allViews,
            final Interval cropBB, // in WORLD coordinates consistent with regs
            final int interpolation,
            final boolean colorByIllumination
    ) {
        final Map<ViewId, ? extends BasicViewDescription<ViewSetup>> vds =
                spimData.getSequenceDescription().getViewDescriptions();

        // 1) Find which views really overlap this crop
        final ArrayList<ViewId> overlapping = LazyFusionTools.overlappingViewIds(
                cropBB,
                allViews,
                regs,
                LazyFusionTools.assembleDimensions(allViews, vds),
                LazyFusionTools.defaultAffineExpansion // small expansion to be conservative
        );

        // 2) Prepare transformed sources for each overlapping view (mapped into the crop)
        final List<RandomAccessibleInterval<FloatType>> transformedSources = new ArrayList<>(overlapping.size());
        final List<Integer> sourceIlluminationIds = new ArrayList<>(overlapping.size());
        for (final ViewId vid : overlapping) {
            final AffineTransform3D model = regs.get(vid).copy();

            final double[] usedDownsampleFactors = new double[3];
            RandomAccessibleInterval<FloatType> input = DownsampleTools.openDownsampled(
                    spimData.getSequenceDescription().getImgLoader(),
                    vid,
                    model,
                    usedDownsampleFactors
            );

            // Transform into the crop’s coordinate frame (and slice to the crop)
            RandomAccessibleInterval<FloatType> transformed = TransformView.transformView(
                    input, model, cropBB, 0, interpolation);

            // Ensure zero-min to match our output interval
            transformed = Views.zeroMin(transformed);
            transformedSources.add(transformed);

            final BasicViewDescription<ViewSetup> vd = vds.get(vid);
            sourceIlluminationIds.add(illuminationIdOf(vd));
        }

        final int n = transformedSources.size();
        
        // Determine whether to color by view ID (regular affine case) or illumination ID (split affine case)
        final int[] palette;
        final int[] colorIndexForSource = new int[n];
        if (colorByIllumination) {
            final LinkedHashMap<Integer, Integer> illuminationColorIndex = new LinkedHashMap<>();
            for (final Integer illuminationId : sourceIlluminationIds)
                illuminationColorIndex.computeIfAbsent(illuminationId, ignored -> illuminationColorIndex.size());
            palette = distinctPalette(illuminationColorIndex.size());
            for (int idx = 0; idx < n; idx++)
                colorIndexForSource[idx] = illuminationColorIndex.get(sourceIlluminationIds.get(idx));
        } else {
            palette = distinctPalette(n);
            for (int idx = 0; idx < n; idx++)
                colorIndexForSource[idx] = idx;
        }

        final Interval outInterval = FusionTools.getFusedZeroMinInterval(cropBB);
        return new OverlayCompositor(outInterval, transformedSources, palette, colorIndexForSource);
    }

    /** Map baseColor (ARGB, with 8-bit RGB) to a color whose RGB channels equal baseColor*level/255 and alpha=255. */
    static int tint(final int baseColor, final int level /*0..255*/) {
        final int br = (baseColor >> 16) & 0xff;
        final int bg = (baseColor >> 8)  & 0xff;
        final int bb = (baseColor)       & 0xff;
        final int r = (br * level) / 255;
        final int g = (bg * level) / 255;
        final int b = (bb * level) / 255;
        return (0xff << 24) | (r << 16) | (g << 8) | b; // opaque
    }

    /** Per-channel max blend (ignores alpha for simplicity; keeps overlays bright). */
    static int maxBlend(final int dst, final int src) {
        final int r = Math.max((dst >> 16) & 0xff, (src >> 16) & 0xff);
        final int g = Math.max((dst >> 8)  & 0xff, (src >> 8)  & 0xff);
        final int b = Math.max(dst & 0xff,        src & 0xff);
        return (0xff << 24) | (r << 16) | (g << 8) | b;
    }

    static int clampTo8bit(final float v) {
        return (v <= 0) ? 0 : (v >= 255f ? 255 : (int) v);
    }

    private static void exportCropAsOmeZarr(
            final OverlayCompositor overlay,
            final String name
    ) throws IOException {
        final Path outputFolder = OME_ZARR_ROOT.resolve(name).normalize();
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
                writeOmeNgffMetadata(writer, pyramid, containerPath.getFileName().toString());
            }
        } finally {
            shutdownExecutor(executor);
        }

        System.out.println("Exported multiscale OME-Zarr crop to " + containerPath);
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
            final String datasetName
    ) throws IOException {
        final Function<Integer, AffineTransform3D> levelToTransform =
                level -> MipmapTransforms.getMipmapTransformDefault(pyramid[level].absoluteDownsamplingDouble());

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

        OverlayCompositor(
                final Interval interval,
                final List<RandomAccessibleInterval<FloatType>> sources,
                final int[] palette,
                final int[] colorIndexForSource
        ) {
            this.sources = new ArrayList<>(sources);
            this.palette = palette;
            this.colorIndexForSource = colorIndexForSource.clone();
            this.spatialDims = new long[3];
            interval.dimensions(this.spatialDims);
        }

        long[] spatialDimensions() {
            return spatialDims.clone();
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
                            final int level = clampTo8bit((val / DISPLAY_MAX) * 255f);
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

    private static int[] defaultBlockSize(final long[] spatialDims) {
        return new int[] {
                blockSizeFor(spatialDims[0]),
                blockSizeFor(spatialDims[1]),
                blockSizeFor(spatialDims[2]),
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
        for (int level = 1; level <= MAX_DOWNSAMPLING_LEVELS; level++) {
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

    /** Simple distinct palette */
    static int[] distinctPalette(final int n) {
        final int[] base = new int[] {
                0xFF0000, // red
                0x00FF00, // green
                0x0000FF, // blue
                0xFFFF00, // yellow
                0xFF00FF, // magenta
                0x00FFFF, // cyan
                0xFFA500, // orange
                0x8000FF, // purple
                0x00FF80, // spring green
                0xFF0080  // pink
        };
        final int[] out = new int[n];
        for (int i = 0; i < n; i++) out[i] = base[i % base.length];
        return out;
    }
}
