package net.preibisch.mvrecon.process.fusion.lazy;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ij.IJ;
import ij.ImagePlus;
import mpicbg.spim.data.generic.sequence.BasicViewDescription;
import mpicbg.spim.data.sequence.ViewDescription;
import mpicbg.spim.data.sequence.ViewId;
import mpicbg.spim.data.sequence.ViewSetup;
import net.imglib2.*;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.ARGBType;
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
import org.slf4j.LoggerFactory;

public class TestColorOverlayOverlaps {

    // Display scaling: map FloatType intensities to 0..255 before tinting.
    static final float DISPLAY_MAX = 100f;

    // Choose interpolation: 1=linear, 0=nearest
    static final int INTERPOLATION = 1;

    public static void main(String[] args) throws Exception {
        final Logger rootLogger = (Logger) LoggerFactory.getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME);
        rootLogger.setLevel(Level.INFO);

        final String xml = "/root/capsule/data/bigstitcher_affine.xml";
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
        final double downsampling = 16;  // or Double.NaN
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
    ) {
        for (int i = 0; i < illuminationBounds.size(); i++) {
            for (int j = i + 1; j < illuminationBounds.size(); j++) {
                final IlluminationBounds tile1 = illuminationBounds.get(i);
                final IlluminationBounds tile2 = illuminationBounds.get(j);

                final Interval overlap = Intervals.intersect(tile1.interval, tile2.interval);
                if (Intervals.isEmpty(overlap)) {
                    continue;
                }

                final RandomAccessibleInterval<ARGBType> argbCrop = overlayCropARGB(
                        spimData,
                        regs,
                        views,
                        overlap,
                        INTERPOLATION,
                        true
                );

                final String name = "overlay_tile" + tile1.illuminationId + "_tile" + tile2.illuminationId;
                ImagePlus imp = net.imglib2.img.display.imagej.ImageJFunctions.wrap(argbCrop, name);
                IJ.saveAsTiff(imp, "/results/" + name + ".tif");

                System.out.println("Created ARGB overlay crop for tiles (" + tile1.illuminationId + "," + tile2.illuminationId + ") "
                        + Arrays.toString(argbCrop.dimensionsAsLongArray()));
            }
        }
    }

    private static void processPerViewOverlaps(
            final SpimData2 spimData,
            final Map<ViewId, AffineTransform3D> regs,
            final List<ViewId> views,
            final Map<ViewId, ? extends BasicViewDescription<ViewSetup>> viewDescriptions
    ) {
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

                final RandomAccessibleInterval<ARGBType> argbCrop = overlayCropARGB(
                        spimData,
                        regs,
                        views,
                        overlap,
                        INTERPOLATION,
                        false
                );

                final String name = "overlay_v" + v1.getViewSetupId() + "_v" + v2.getViewSetupId();
                ImagePlus imp = net.imglib2.img.display.imagej.ImageJFunctions.wrap(argbCrop, name);
                IJ.saveAsTiff(imp, "/results/affine/" + name + ".tif");

                System.out.println("Created ARGB overlay crop for views (" + v1.getViewSetupId() + "," + v2.getViewSetupId() + ") "
                        + Arrays.toString(argbCrop.dimensionsAsLongArray()));
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

    /** Create a color overlay ARGB crop over the given bounding box.
     *  It auto-selects only the views that overlap the crop and tints each differently. */
    static RandomAccessibleInterval<ARGBType> overlayCropARGB(
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
        if (overlapping.isEmpty()) {
            // Return an empty ARGB image of the crop size (all zeros)
            return emptyARGB(FusionTools.getFusedZeroMinInterval(cropBB));
        }

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

        // 4) Allocate ARGB output for the crop (zero-min with same dims as crop)
        final Interval outInterval = FusionTools.getFusedZeroMinInterval(cropBB);
        final long[] dims = new long[outInterval.numDimensions()];
        outInterval.dimensions(dims);
        final ArrayImg<ARGBType, ?> out = ArrayImgs.argbs(dims);

        // 5) Composite: per-pixel, tint each source and max-blend into output
        final Cursor<ARGBType> cOut = Views.flatIterable(out).localizingCursor();
        final List<RandomAccess<FloatType>> ras = new ArrayList<>(n);
        for (RandomAccessibleInterval<FloatType> src : transformedSources)
            ras.add(src.randomAccess());

        final int[] pos = new int[dims.length];
        while (cOut.hasNext()) {
            final ARGBType outPix = cOut.next();
            cOut.localize(pos); // 3D voxel position in zero-min crop coords

            int dst = 0; // ARGB 0 == fully transparent black
            for (int k = 0; k < n; k++) {
                final RandomAccess<FloatType> ra = ras.get(k);
                ra.setPosition(pos);

                final float val = ra.get().getRealFloat();
                if (val <= 0) continue; // skip zero

                // Scale intensity into 0..255 (tweak DISPLAY_MAX to expected data range)
                final int level = clampTo8bit((val / DISPLAY_MAX) * 255f);
                // System.out.println(level);

                // Apply per-illumination tint with intensity "level"
                final int src = tint(palette[colorIndexForSource[k]], level);

                // Per-channel max blend (good for alignment inspection)
                dst = maxBlend(dst, src);
            }
            outPix.set(dst);
        }

        return out;
    }

    /** Create an empty ARGB image for the given zero-min interval. */
    static RandomAccessibleInterval<ARGBType> emptyARGB(final Interval zeroMinInterval) {
        final long[] dims = new long[zeroMinInterval.numDimensions()];
        zeroMinInterval.dimensions(dims);
        return ArrayImgs.argbs(dims);
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
