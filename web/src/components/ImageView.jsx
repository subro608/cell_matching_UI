// One image panel: Viv multiscale image + mask outlines + markers + lines, in native image pixels (OrthographicView, y down).
// OverlayCanvas is a second, click-through canvas stacked on a panel (IV warped onto EX) so swipe / checkerboard / blend are CSS.
import { useEffect, useMemo, useRef, useState } from 'react';
import DeckGL from '@deck.gl/react';
import { OrthographicView } from '@deck.gl/core';
import { LineLayer, PolygonLayer, ScatterplotLayer, TextLayer } from '@deck.gl/layers';
import { MultiscaleImageLayer } from '@hms-dbmi/viv';
import { getOutlines, hasPlane } from '../lib/data';

const VIEW = new OrthographicView({ id: 'ortho', controller: true });
const STATIC_VIEW = new OrthographicView({ id: 'ortho', controller: false });

export function fitView(size, box) {
  if (!box.width || !size.width) return null;
  const zoom = Math.log2(Math.min(size.width / box.width, size.height / box.height) * 0.95);
  return { target: [box.width / 2, box.height / 2, 0], zoom };
}

export function useOutlines(src, sets, visible, z) {
  const [data, setData] = useState({});
  useEffect(() => {
    let live = true;
    const on = sets.filter((s) => visible[s.name] && hasPlane(s, z));
    Promise.all(on.map((s) => getOutlines(src, s, z).then((rows) => [s.name, rows])))
      .then((pairs) => { if (live) setData(Object.fromEntries(pairs)); })
      .catch((e) => console.error(e));
    return () => { live = false; };
  }, [src, sets, visible, z]);
  return data;
}

function useSize(ref) {
  const [size, setSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    const ro = new ResizeObserver(([e]) => setSize({ width: e.contentRect.width, height: e.contentRect.height }));
    if (ref.current) ro.observe(ref.current);
    return () => ro.disconnect();
  }, [ref]);
  return size;
}

export default function ImageView({
  title, src, image, selection, contrast, colors, channelsVisible, maskSets, maskVisible, z = 0, viewState, onViewState,
  onClick, highlight = [], markers = [], lines = [], only = null, children,
}) {
  const ref = useRef(null);
  const size = useSize(ref);
  const outlines = useOutlines(src, maskSets, maskVisible, z);

  const layers = useMemo(() => {
    const out = [new MultiscaleImageLayer({
      id: `${title}-image`, loader: image.data, selections: selection, contrastLimits: contrast, colors, channelsVisible,
    })];
    for (const s of maskSets) {
      let rows = outlines[s.name];
      if (!rows || !maskVisible[s.name]) continue;
      if (only?.[s.name]) rows = rows.filter((d) => only[s.name].has(d.id));        // agreement filter
      out.push(new PolygonLayer({
        id: `${title}-mask-${s.name}-${only ? 'f' : 'a'}`, data: rows, getPolygon: (d) => d.polygon, stroked: true, filled: true,
        getFillColor: [0, 0, 0, 0], getLineColor: [...(s.color || [255, 255, 0]), 200], lineWidthUnits: 'pixels', getLineWidth: 1,
        pickable: true, autoHighlight: true, highlightColor: [255, 255, 255, 60],
      }));
    }
    for (const h of highlight) {
      const rows = (outlines[h.set] || []).filter((d) => h.ids.has(d.id));
      if (rows.length) out.push(new PolygonLayer({
        id: `${title}-hl-${h.set}-${h.key}`, data: rows, getPolygon: (d) => d.polygon, stroked: true, filled: false,
        getLineColor: h.color, lineWidthUnits: 'pixels', getLineWidth: h.width || 3,
      }));
    }
    if (lines.length) {
      out.push(new LineLayer({
        id: `${title}-lines`, data: lines, getSourcePosition: (d) => d.from, getTargetPosition: (d) => d.to, getColor: (d) => d.color,
        getWidth: (d) => d.w || 1.5, widthUnits: 'pixels',
      }));
    }
    if (markers.length) {
      out.push(new ScatterplotLayer({
        id: `${title}-markers`, data: markers, getPosition: (d) => [d.x, d.y], getRadius: (d) => d.r || 6, radiusUnits: 'pixels',
        stroked: true, filled: false, getLineColor: (d) => d.color, lineWidthUnits: 'pixels', getLineWidth: (d) => d.w || 2, pickable: true,
      }));
      out.push(new TextLayer({
        id: `${title}-labels`, data: markers.filter((d) => d.label), getPosition: (d) => [d.x, d.y], getText: (d) => d.label,
        getColor: (d) => d.color, getSize: 13, getPixelOffset: [10, -10], fontWeight: 700, outlineWidth: 2, outlineColor: [0, 0, 0, 255],
        fontSettings: { sdf: true },
      }));
    }
    return out;
  }, [title, image, selection, contrast, colors, channelsVisible, maskSets, maskVisible, outlines, highlight, markers, lines, only]);

  return (
    <div className="view" ref={ref}>
      <div className="view-title">{title}</div>
      {size.width > 0 && viewState && (
        <DeckGL
          views={[VIEW]} viewState={{ ortho: viewState }} layers={layers} width={size.width} height={size.height}
          onViewStateChange={({ viewState: v }) => onViewState({ target: v.target, zoom: v.zoom })}
          onClick={(info) => info.coordinate && onClick && onClick({ x: info.coordinate[0], y: info.coordinate[1], object: info.object, layer: info.layer?.id })}
          getCursor={({ isDragging, isHovering }) => (isDragging ? 'grabbing' : isHovering ? 'pointer' : 'crosshair')}
        />
      )}
      {children}
    </div>
  );
}

// IV drawn through modelMatrix (native IV px -> EX px) on top of an EX panel. mode: blend | swipe | checker; amount 0..1.
export function OverlayCanvas({ image, selection, contrast, colors, matrix, viewState, mode, amount, cell = 64 }) {
  const ref = useRef(null);
  const size = useSize(ref);
  const layers = useMemo(() => [new MultiscaleImageLayer({
    id: 'overlay-iv', loader: image.data, selections: selection, contrastLimits: contrast, colors, channelsVisible: selection.map(() => true), modelMatrix: matrix,
  })], [image, selection, contrast, colors, matrix]);
  const style = { position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 1 };
  if (mode === 'blend') Object.assign(style, { mixBlendMode: 'screen', opacity: amount });
  if (mode === 'swipe') style.clipPath = `inset(0 0 0 ${(amount * 100).toFixed(1)}%)`;
  if (mode === 'checker') {
    const m = `repeating-conic-gradient(#000 0 25%, transparent 0 50%) 0 0 / ${cell * 2}px ${cell * 2}px`;
    Object.assign(style, { maskImage: m, WebkitMaskImage: m, maskRepeat: 'repeat' });
  }
  return (
    <div ref={ref} style={style}>
      {size.width > 0 && viewState && <DeckGL views={[STATIC_VIEW]} viewState={{ ortho: viewState }} layers={layers} width={size.width} height={size.height} />}
    </div>
  );
}
