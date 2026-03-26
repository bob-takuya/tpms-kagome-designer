import * as THREE from 'three';
import type { Strip, StripSegment, Junction } from './kagome';
import type { HalfEdgeMesh } from './halfEdge';

export interface UnfoldedStrip {
  stripId: string;
  family: number;
  layer: number;
  segments: UnfoldedSegment[];
  boundingBox: { minX: number; maxX: number; minY: number; maxY: number };
}

export interface UnfoldedSegment {
  startJunctionId: number | null;
  endJunctionId: number | null;
  leftBoundary: THREE.Vector2[];
  rightBoundary: THREE.Vector2[];
  centerline: THREE.Vector2[];
  holes: UnfoldedHole[];
  width: number;
}

export interface UnfoldedHole {
  junctionId: number;
  center: THREE.Vector2;
  radius: number;
}

// Sequential triangle unfolding to preserve edge lengths
export function unfoldStrip(
  strip: Strip,
  mesh: HalfEdgeMesh,
  junctions: Junction[],
  scale: number
): UnfoldedStrip {
  const unfoldedSegments: UnfoldedSegment[] = [];

  for (const segment of strip.segments) {
    const unfolded = unfoldSegment(segment, strip, mesh, junctions, scale);
    unfoldedSegments.push(unfolded);
  }

  // Calculate bounding box
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;

  for (const seg of unfoldedSegments) {
    for (const pt of [...seg.leftBoundary, ...seg.rightBoundary, ...seg.centerline]) {
      minX = Math.min(minX, pt.x);
      maxX = Math.max(maxX, pt.x);
      minY = Math.min(minY, pt.y);
      maxY = Math.max(maxY, pt.y);
    }
  }

  return {
    stripId: strip.id,
    family: strip.family,
    layer: strip.layer,
    segments: unfoldedSegments,
    boundingBox: { minX, maxX, minY, maxY }
  };
}

function unfoldSegment(
  segment: StripSegment,
  _strip: Strip,
  _mesh: HalfEdgeMesh,
  junctions: Junction[],
  scale: number
): UnfoldedSegment {
  const centerline2D: THREE.Vector2[] = [];
  const leftBoundary2D: THREE.Vector2[] = [];
  const rightBoundary2D: THREE.Vector2[] = [];
  const holes: UnfoldedHole[] = [];

  if (segment.points.length < 2) {
    return {
      startJunctionId: segment.startJunctionId,
      endJunctionId: segment.endJunctionId,
      leftBoundary: [],
      rightBoundary: [],
      centerline: [],
      holes: [],
      width: segment.width * scale
    };
  }

  // Initialize with first point at origin
  let currentPos = new THREE.Vector2(0, 0);
  let currentAngle = 0;
  let accumulatedLength = 0;

  centerline2D.push(currentPos.clone());

  // Compute perpendicular offset for boundaries
  const halfWidth = (segment.width * scale) / 2;

  // Add first boundary points
  const firstDir2D = new THREE.Vector2(Math.cos(currentAngle), Math.sin(currentAngle));
  const firstPerp = new THREE.Vector2(-firstDir2D.y, firstDir2D.x);
  leftBoundary2D.push(currentPos.clone().add(firstPerp.clone().multiplyScalar(halfWidth)));
  rightBoundary2D.push(currentPos.clone().sub(firstPerp.clone().multiplyScalar(halfWidth)));

  // Unfold each segment preserving edge lengths
  for (let i = 1; i < segment.points.length; i++) {
    const p0 = segment.points[i - 1];
    const p1 = segment.points[i];

    // 3D edge length
    const edgeLength3D = p0.distanceTo(p1) * scale;

    // Place next point along current direction
    currentPos = currentPos.clone().add(
      new THREE.Vector2(
        Math.cos(currentAngle) * edgeLength3D,
        Math.sin(currentAngle) * edgeLength3D
      )
    );

    centerline2D.push(currentPos.clone());
    accumulatedLength += edgeLength3D;

    // Compute tangent direction for boundaries
    if (i < segment.points.length - 1) {
      const p2 = segment.points[i + 1];
      const dir1 = new THREE.Vector3().subVectors(p1, p0).normalize();
      const dir2 = new THREE.Vector3().subVectors(p2, p1).normalize();

      // Angle change between segments
      const angleChange = Math.atan2(
        dir1.clone().cross(dir2).length(),
        dir1.dot(dir2)
      );

      // Update direction (turning gradually)
      currentAngle += angleChange * 0.5;
    }

    const dir2D = new THREE.Vector2(Math.cos(currentAngle), Math.sin(currentAngle));
    const perp = new THREE.Vector2(-dir2D.y, dir2D.x);
    leftBoundary2D.push(currentPos.clone().add(perp.clone().multiplyScalar(halfWidth)));
    rightBoundary2D.push(currentPos.clone().sub(perp.clone().multiplyScalar(halfWidth)));
  }

  // Map junction holes to 2D
  for (const junction of junctions) {
    const isStartJunction = segment.startJunctionId === junction.id;
    const isEndJunction = segment.endJunctionId === junction.id;

    if (isStartJunction) {
      holes.push({
        junctionId: junction.id,
        center: centerline2D[0].clone(),
        radius: junction.holeRadius * scale
      });
    } else if (isEndJunction) {
      holes.push({
        junctionId: junction.id,
        center: centerline2D[centerline2D.length - 1].clone(),
        radius: junction.holeRadius * scale
      });
    } else {
      // Check if junction is along this segment
      const junctionPos3D = junction.position;
      let minDist = Infinity;
      let closestIdx = -1;

      for (let i = 0; i < segment.points.length; i++) {
        const dist = segment.points[i].distanceTo(junctionPos3D);
        if (dist < minDist) {
          minDist = dist;
          closestIdx = i;
        }
      }

      // If junction is close to segment
      const threshold = segment.width * 2;
      if (minDist < threshold && closestIdx >= 0 && closestIdx < centerline2D.length) {
        holes.push({
          junctionId: junction.id,
          center: centerline2D[closestIdx].clone(),
          radius: junction.holeRadius * scale
        });
      }
    }
  }

  return {
    startJunctionId: segment.startJunctionId,
    endJunctionId: segment.endJunctionId,
    leftBoundary: leftBoundary2D,
    rightBoundary: rightBoundary2D,
    centerline: centerline2D,
    holes,
    width: segment.width * scale
  };
}

// Layout multiple unfolded strips on a 2D canvas
export interface StripLayout {
  strips: UnfoldedStrip[];
  totalWidth: number;
  totalHeight: number;
  positions: THREE.Vector2[];
}

export function layoutStrips(
  strips: UnfoldedStrip[],
  margin: number
): StripLayout {
  const positions: THREE.Vector2[] = [];
  let currentX = margin;
  let maxHeight = 0;

  for (const strip of strips) {
    const width = strip.boundingBox.maxX - strip.boundingBox.minX;
    const height = strip.boundingBox.maxY - strip.boundingBox.minY;

    // Offset to move strip's min corner to currentX, margin
    positions.push(new THREE.Vector2(
      currentX - strip.boundingBox.minX,
      margin - strip.boundingBox.minY
    ));

    currentX += width + margin;
    maxHeight = Math.max(maxHeight, height);
  }

  return {
    strips,
    totalWidth: currentX,
    totalHeight: maxHeight + 2 * margin,
    positions
  };
}

// Apply layout offset to unfolded strips
export function applyLayout(layout: StripLayout): UnfoldedStrip[] {
  return layout.strips.map((strip, idx) => {
    const offset = layout.positions[idx];

    return {
      ...strip,
      segments: strip.segments.map(seg => ({
        ...seg,
        leftBoundary: seg.leftBoundary.map(p => p.clone().add(offset)),
        rightBoundary: seg.rightBoundary.map(p => p.clone().add(offset)),
        centerline: seg.centerline.map(p => p.clone().add(offset)),
        holes: seg.holes.map(h => ({
          ...h,
          center: h.center.clone().add(offset)
        }))
      })),
      boundingBox: {
        minX: strip.boundingBox.minX + offset.x,
        maxX: strip.boundingBox.maxX + offset.x,
        minY: strip.boundingBox.minY + offset.y,
        maxY: strip.boundingBox.maxY + offset.y
      }
    };
  });
}
