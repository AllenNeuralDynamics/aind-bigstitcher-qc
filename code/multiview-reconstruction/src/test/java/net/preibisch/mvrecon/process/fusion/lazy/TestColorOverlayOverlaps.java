package net.preibisch.mvrecon.process.fusion.lazy;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ij.IJ;
import ij.ImagePlus;
import mpicbg.spim.data.generic.sequence.BasicViewDescription;
import mpicbg.spim.data.sequence.ViewDescription;
import mpicbg.spim.data.sequence.ViewId;
import net.imglib2.*;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.ARGBType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import net.imglib2.realtransform.AffineTransform3D;
import net.preibisch.mvrecon.process.fusion.transformed.TransformVirtual;
import net.preibisch.mvrecon.fiji.plugin.fusion.FusionGUI.FusionType;
import net.preibisch.mvrecon.fiji.spimdata.SpimData2;
import net.preibisch.mvrecon.fiji.spimdata.XmlIoSpimData2;
import net.preibisch.mvrecon.process.downsampling.DownsampleTools;
import net.preibisch.mvrecon.process.fusion.FusionTools;
import net.preibisch.mvrecon.process.fusion.transformed.TransformView;
import net.preibisch.mvrecon.process.fusion.lazy.LazyFusionTools;
import net.preibisch.mvrecon.process.fusion.lazy.LazyAffineFusion;
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

        // Pairwise search to propose overlap candidates
        for (int i = 0; i < views.size(); i++) {
            for (int j = i + 1; j < views.size(); j++) {
                final ViewId v1 = views.get(i);
                final ViewId v2 = views.get(j);

                final Dimensions s1 = spimData.getSequenceDescription().getViewDescriptions().get(v1).getViewSetup().getSize();
                final Dimensions s2 = spimData.getSequenceDescription().getViewDescriptions().get(v2).getViewSetup().getSize();

                final Interval bb1 = Intervals.largestContainedInterval(regs.get(v1).estimateBounds(new FinalInterval(s1)));
                final Interval bb2 = Intervals.largestContainedInterval(regs.get(v2).estimateBounds(new FinalInterval(s2)));

                final Interval overlap = Intervals.intersect(bb1, bb2);
                if (Intervals.isEmpty(overlap)) {
                    // No real overlap on at least one axis — skip.
                    continue;
                }

                // Make a multi-view ARGB overlay for THIS overlap region
                final RandomAccessibleInterval<ARGBType> argbCrop = overlayCropARGB(
                        spimData,
                        regs,
                        views,
                        overlap,
                        INTERPOLATION
                );

                // Save crop as Tiff
                final String name = "overlay_v" + v1.getViewSetupId() + "_v" + v2.getViewSetupId();
                ImagePlus imp = net.imglib2.img.display.imagej.ImageJFunctions.wrap(argbCrop, name);
                IJ.saveAsTiff(imp, "/results/" + name + ".tif");

                System.out.println("Created ARGB overlay crop for (" + v1.getViewSetupId() + "," + v2.getViewSetupId() + ") "
                        + Arrays.toString(argbCrop.dimensionsAsLongArray()));
            }
        }
    }

    /** Create a color overlay ARGB crop over the given bounding box.
     *  It auto-selects only the views that overlap the crop and tints each differently. */
    static RandomAccessibleInterval<ARGBType> overlayCropARGB(
            final SpimData2 spimData,
            final Map<ViewId, AffineTransform3D> regs,
            final Collection<? extends ViewId> allViews,
            final Interval cropBB, // in WORLD coordinates consistent with regs
            final int interpolation
    ) {
        final Map<ViewId, ? extends BasicViewDescription<?>> vds =
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
        }

        // 3) Build a distinct color palette (RGBA ints) for N views
        final int n = transformedSources.size();
        final int[] palette = distinctPalette(n);

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

                // Apply per-view tint with intensity "level"
                final int src = tint(palette[k], level);

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
