package org.aind.bigstitcher.qc;

import bdv.util.MipmapTransforms;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.ByteArray;
import net.imglib2.realtransform.AffineTransform3D;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.preibisch.mvrecon.process.n5api.N5ApiTools;
import net.preibisch.mvrecon.process.n5api.N5ApiTools.MultiResolutionLevelInfo;
import net.preibisch.mvrecon.fiji.spimdata.imgloaders.OMEZarrAttibutes;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.janelia.saalfeldlab.n5.universe.metadata.ome.ngff.v04.OmeNgffMultiScaleMetadata;

final class ZarrUtils {

    private ZarrUtils() {
    }

    static int[] defaultBlockSize(final long[] spatialDims, final Integer userDefinedBlockSize) {
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
        return (int) Math.max(1, Math.min(dim, 128));
    }

    static int[][] defaultDownsamplings(final long[] spatialDims, final int maxDownsamplingLevels) {
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

    static void writeS0Blocks(
            final CompositeUtils.BlockRenderer overlay,
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
                long t0 = System.currentTimeMillis();
                overlay.renderBlock(blockMin, blockImg);
                long t1 = System.currentTimeMillis();
                // System.out.println(String.format("Rendered block at %d,%d,%d in %d ms",
                //         blockMin[0], blockMin[1], blockMin[2], (t1 - t0)));

                final long[] gridOffset = new long[] { blockCopy[2][0], blockCopy[2][1], blockCopy[2][2], 0, 0 };
                N5Utils.saveNonEmptyBlock(blockImg, writer, levelInfo.dataset, gridOffset, new UnsignedByteType());
                return null;
            }));
        }

        waitForFutures(futures);
    }

    static void writeDownsampledLevels(
            final N5Writer writer,
            final MultiResolutionLevelInfo[] pyramid,
            final ExecutorService executor
    ) {
        if (pyramid.length <= 1)
            return;

        for (int level = 1; level < pyramid.length; level++) {
            final MultiResolutionLevelInfo current = pyramid[level];
            final MultiResolutionLevelInfo previous = pyramid[level - 1];

            final List<long[][]> grid = N5ApiTools.assembleJobs(current.dimensions, current.blockSize);
            final List<Future<?>> futures = new ArrayList<>();

            for (long[][] gridBlock : grid) {
                final long[][] blockCopy = cloneGridBlock(gridBlock);
                futures.add(executor.submit(() -> {
                    N5ApiTools.writeDownsampledBlock(writer, current, previous, blockCopy);
                }));
            }

            waitForFutures(futures);
        }
    }

    static void writeOmeNgffMetadata(
            final N5Writer writer,
            final MultiResolutionLevelInfo[] pyramid,
            final String datasetName,
            final double[] worldMin3d,
            final String unit
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
                unit,
                pyramid.length,
                level -> "/" + level,
                levelToTransform
        );

        writer.setAttribute("/", "multiscales", metadata);
    }

    static double[] computeLevelTranslation(final double[] worldMin3d, final double[] absoluteDownsampling) {
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

    static void shutdownExecutor(final ExecutorService executor) {
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

    private static void waitForFutures(final List<Future<?>> futures) {
        for (Future<?> future : futures) {
            try {
                future.get();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Interrupted while processing OME-Zarr blocks", e);
            } catch (java.util.concurrent.ExecutionException e) {
                // Log full stack traces for both the wrapper and the original cause
                System.err.println("Executor task failed. Printing full stack trace:");
                e.printStackTrace();
                final Throwable cause = e.getCause();
                if (cause != null) {
                    System.err.println("Caused by:");
                    cause.printStackTrace();
                }
                // Rethrow with the ExecutionException to preserve the full chain
                throw new RuntimeException("Failed to process OME-Zarr blocks", e);
            }
        }
    }

    private static long[][] cloneGridBlock(final long[][] gridBlock) {
        final long[][] copy = new long[gridBlock.length][];
        for (int i = 0; i < gridBlock.length; i++)
            copy[i] = gridBlock[i].clone();
        return copy;
    }
}
