import * as THREE from 'three';
import type { UnfoldedStrip } from '../core/unfold';

export interface SVGExportOptions {
  includeHoleIds: boolean;
  includeFoldLines: boolean;
  strokeWidth: number;
}

const FAMILY_COLORS = ['#ff4444', '#ffff44', '#44ff44']; // Red, Yellow, Green

export function generateSVG(
  strips: UnfoldedStrip[],
  width: number,
  height: number,
  options: SVGExportOptions
): string {
  const { includeHoleIds, includeFoldLines, strokeWidth } = options;

  const elements: string[] = [];

  // SVG header
  elements.push(`<?xml version="1.0" encoding="UTF-8"?>`);
  elements.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`);

  // Style definitions
  elements.push(`<defs>`);
  elements.push(`  <style>`);
  elements.push(`    .strip-a { stroke: ${FAMILY_COLORS[0]}; fill: none; stroke-width: ${strokeWidth}; }`);
  elements.push(`    .strip-b { stroke: ${FAMILY_COLORS[1]}; fill: none; stroke-width: ${strokeWidth}; }`);
  elements.push(`    .strip-c { stroke: ${FAMILY_COLORS[2]}; fill: none; stroke-width: ${strokeWidth}; }`);
  elements.push(`    .hole { stroke: #00ffff; fill: none; stroke-width: ${strokeWidth}; }`);
  elements.push(`    .fold-line { stroke: #ff00ff; fill: none; stroke-width: ${strokeWidth * 0.5}; stroke-dasharray: 4,2; }`);
  elements.push(`    .label { font-family: Arial, sans-serif; font-size: 8px; fill: #ffffff; }`);
  elements.push(`    .hole-id { font-family: Arial, sans-serif; font-size: 6px; fill: #00ffff; text-anchor: middle; }`);
  elements.push(`  </style>`);
  elements.push(`</defs>`);

  // Background
  elements.push(`<rect width="100%" height="100%" fill="#1a1a1a"/>`);

  const familyClasses = ['strip-a', 'strip-b', 'strip-c'];

  for (const strip of strips) {
    const className = familyClasses[strip.family];

    // Group for this strip
    elements.push(`<g id="strip-${strip.stripId}">`);

    for (const segment of strip.segments) {
      // Left boundary
      if (segment.leftBoundary.length > 1) {
        elements.push(createPolyline(segment.leftBoundary, className, height));
      }

      // Right boundary
      if (segment.rightBoundary.length > 1) {
        elements.push(createPolyline(segment.rightBoundary, className, height));
      }

      // End caps
      if (segment.leftBoundary.length > 0 && segment.rightBoundary.length > 0) {
        elements.push(createLine(
          segment.leftBoundary[0],
          segment.rightBoundary[0],
          className,
          height
        ));
        elements.push(createLine(
          segment.leftBoundary[segment.leftBoundary.length - 1],
          segment.rightBoundary[segment.rightBoundary.length - 1],
          className,
          height
        ));
      }

      // Fold lines
      if (includeFoldLines && segment.centerline.length > 1) {
        elements.push(createPolyline(segment.centerline, 'fold-line', height));
      }

      // Holes
      for (const hole of segment.holes) {
        elements.push(createCircle(hole.center, hole.radius, 'hole', height));

        if (includeHoleIds) {
          elements.push(createText(
            hole.center,
            String(hole.junctionId),
            'hole-id',
            height
          ));
        }
      }
    }

    // Strip label
    if (strip.segments.length > 0 && strip.segments[0].centerline.length > 0) {
      const labelPos = strip.segments[0].centerline[0].clone();
      labelPos.y += strip.segments[0].width * 0.7;
      elements.push(createText(labelPos, strip.stripId, 'label', height));
    }

    elements.push(`</g>`);
  }

  elements.push(`</svg>`);

  return elements.join('\n');
}

function createPolyline(points: THREE.Vector2[], className: string, svgHeight: number): string {
  const pointsStr = points.map(p => `${p.x.toFixed(2)},${(svgHeight - p.y).toFixed(2)}`).join(' ');
  return `  <polyline points="${pointsStr}" class="${className}"/>`;
}

function createLine(
  p1: THREE.Vector2,
  p2: THREE.Vector2,
  className: string,
  svgHeight: number
): string {
  return `  <line x1="${p1.x.toFixed(2)}" y1="${(svgHeight - p1.y).toFixed(2)}" x2="${p2.x.toFixed(2)}" y2="${(svgHeight - p2.y).toFixed(2)}" class="${className}"/>`;
}

function createCircle(
  center: THREE.Vector2,
  radius: number,
  className: string,
  svgHeight: number
): string {
  return `  <circle cx="${center.x.toFixed(2)}" cy="${(svgHeight - center.y).toFixed(2)}" r="${radius.toFixed(2)}" class="${className}"/>`;
}

function createText(
  position: THREE.Vector2,
  text: string,
  className: string,
  svgHeight: number
): string {
  return `  <text x="${position.x.toFixed(2)}" y="${(svgHeight - position.y).toFixed(2)}" class="${className}">${text}</text>`;
}
