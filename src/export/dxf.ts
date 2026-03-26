import * as THREE from 'three';
import type { UnfoldedStrip } from '../core/unfold';

export interface DXFLayer {
  name: string;
  color: number; // AutoCAD color index
}

export const DXF_LAYERS: Record<string, DXFLayer> = {
  STRIPS_A: { name: 'STRIPS_A', color: 1 }, // Red
  STRIPS_B: { name: 'STRIPS_B', color: 2 }, // Yellow
  STRIPS_C: { name: 'STRIPS_C', color: 3 }, // Green
  HOLES: { name: 'HOLES', color: 4 }, // Cyan
  HOLE_IDS: { name: 'HOLE_IDS', color: 7 }, // White
  LABELS: { name: 'LABELS', color: 7 }, // White
  FOLD_LINES: { name: 'FOLD_LINES', color: 6 }, // Magenta
};

export function generateDXF(
  strips: UnfoldedStrip[],
  includeHoleIds: boolean,
  includeFoldLines: boolean
): string {
  const lines: string[] = [];

  // DXF Header
  lines.push('0', 'SECTION');
  lines.push('2', 'HEADER');
  lines.push('9', '$ACADVER');
  lines.push('1', 'AC1014');
  lines.push('0', 'ENDSEC');

  // Tables section (layers)
  lines.push('0', 'SECTION');
  lines.push('2', 'TABLES');

  // Layer table
  lines.push('0', 'TABLE');
  lines.push('2', 'LAYER');
  lines.push('70', String(Object.keys(DXF_LAYERS).length));

  for (const layer of Object.values(DXF_LAYERS)) {
    lines.push('0', 'LAYER');
    lines.push('2', layer.name);
    lines.push('70', '0');
    lines.push('62', String(layer.color));
    lines.push('6', 'CONTINUOUS');
  }

  lines.push('0', 'ENDTAB');
  lines.push('0', 'ENDSEC');

  // Entities section
  lines.push('0', 'SECTION');
  lines.push('2', 'ENTITIES');

  const familyLayers = ['STRIPS_A', 'STRIPS_B', 'STRIPS_C'];

  for (const strip of strips) {
    const layerName = familyLayers[strip.family];

    for (const segment of strip.segments) {
      // Left boundary polyline
      if (segment.leftBoundary.length > 1) {
        addPolyline(lines, segment.leftBoundary, layerName);
      }

      // Right boundary polyline
      if (segment.rightBoundary.length > 1) {
        addPolyline(lines, segment.rightBoundary, layerName);
      }

      // End caps (connect left to right at ends)
      if (segment.leftBoundary.length > 0 && segment.rightBoundary.length > 0) {
        // Start cap
        addLine(lines, segment.leftBoundary[0], segment.rightBoundary[0], layerName);
        // End cap
        addLine(
          lines,
          segment.leftBoundary[segment.leftBoundary.length - 1],
          segment.rightBoundary[segment.rightBoundary.length - 1],
          layerName
        );
      }

      // Fold lines (centerline)
      if (includeFoldLines && segment.centerline.length > 1) {
        addPolyline(lines, segment.centerline, 'FOLD_LINES');
      }

      // Holes
      for (const hole of segment.holes) {
        addCircle(lines, hole.center, hole.radius, 'HOLES');

        if (includeHoleIds) {
          addText(lines, hole.center, String(hole.junctionId), 'HOLE_IDS', hole.radius * 0.8);
        }
      }
    }

    // Strip label
    if (strip.segments.length > 0 && strip.segments[0].centerline.length > 0) {
      const labelPos = strip.segments[0].centerline[0].clone();
      labelPos.y += strip.segments[0].width * 0.5;
      addText(lines, labelPos, strip.stripId, 'LABELS', strip.segments[0].width * 0.3);
    }
  }

  lines.push('0', 'ENDSEC');
  lines.push('0', 'EOF');

  return lines.join('\n');
}

function addPolyline(lines: string[], points: THREE.Vector2[], layer: string): void {
  lines.push('0', 'LWPOLYLINE');
  lines.push('8', layer);
  lines.push('90', String(points.length));
  lines.push('70', '0'); // Open polyline

  for (const pt of points) {
    lines.push('10', String(pt.x));
    lines.push('20', String(pt.y));
  }
}

function addLine(lines: string[], p1: THREE.Vector2, p2: THREE.Vector2, layer: string): void {
  lines.push('0', 'LINE');
  lines.push('8', layer);
  lines.push('10', String(p1.x));
  lines.push('20', String(p1.y));
  lines.push('11', String(p2.x));
  lines.push('21', String(p2.y));
}

function addCircle(lines: string[], center: THREE.Vector2, radius: number, layer: string): void {
  lines.push('0', 'CIRCLE');
  lines.push('8', layer);
  lines.push('10', String(center.x));
  lines.push('20', String(center.y));
  lines.push('40', String(radius));
}

function addText(
  lines: string[],
  position: THREE.Vector2,
  text: string,
  layer: string,
  height: number
): void {
  lines.push('0', 'TEXT');
  lines.push('8', layer);
  lines.push('10', String(position.x));
  lines.push('20', String(position.y));
  lines.push('40', String(height));
  lines.push('1', text);
  lines.push('72', '1'); // Center horizontal alignment
  lines.push('11', String(position.x));
  lines.push('21', String(position.y));
}

// Generate junction CSV
export function generateJunctionCSV(
  strips: UnfoldedStrip[]
): string {
  const lines: string[] = ['junction_id,strip_id,x,y,radius'];

  for (const strip of strips) {
    for (const segment of strip.segments) {
      for (const hole of segment.holes) {
        lines.push(`${hole.junctionId},${strip.stripId},${hole.center.x.toFixed(4)},${hole.center.y.toFixed(4)},${hole.radius.toFixed(4)}`);
      }
    }
  }

  return lines.join('\n');
}
