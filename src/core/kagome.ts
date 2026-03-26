import * as THREE from 'three';
import type { HalfEdgeMesh } from './halfEdge';
import type { Isoline } from './connectionLaplacian';

export interface Strip {
  id: string;
  family: number; // 0, 1, 2 for A, B, C
  layer: number; // 0, 1, 2 for over/under assignment
  isolines: [Isoline, Isoline]; // Boundary isolines
  centerline: THREE.Vector3[];
  width: number;
  junctions: Junction[];
  segments: StripSegment[];
}

export interface Junction {
  id: number;
  position: THREE.Vector3;
  stripIds: string[];
  holeRadius: number;
  faceIndex: number;
}

export interface StripSegment {
  startJunctionId: number | null;
  endJunctionId: number | null;
  points: THREE.Vector3[];
  width: number;
  faceIndices: number[];
}

export interface KagomePattern {
  strips: Strip[];
  junctions: Junction[];
  families: [Strip[], Strip[], Strip[]];
}

// Method B: Constant width ratio
export function extractKagomeStrips(
  mesh: HalfEdgeMesh,
  isolinesByFamily: [Isoline[], Isoline[], Isoline[]],
  widthRatio: number,
  holeRadius: number
): KagomePattern {
  const families: [Strip[], Strip[], Strip[]] = [[], [], []];
  const allStrips: Strip[] = [];

  // Create strips from isolines for each family
  for (let family = 0; family < 3; family++) {
    const familyIsolines = isolinesByFamily[family];

    for (let i = 0; i < familyIsolines.length; i++) {
      const isoline1 = familyIsolines[i];
      const isoline2 = familyIsolines[(i + 1) % familyIsolines.length];

      if (isoline1.points.length < 2 || isoline2.points.length < 2) continue;

      // Calculate strip width based on distance between isolines
      const avgWidth = calculateAverageDistance(isoline1.points, isoline2.points) * widthRatio;

      const strip: Strip = {
        id: `${String.fromCharCode(65 + family)}${i + 1}`,
        family,
        layer: family, // Initial layer assignment
        isolines: [isoline1, isoline2],
        centerline: computeCenterline(isoline1.points, isoline2.points),
        width: avgWidth,
        junctions: [],
        segments: []
      };

      families[family].push(strip);
      allStrips.push(strip);
    }
  }

  // Find junctions (intersections between strips from different families)
  const junctions = findJunctions(allStrips, mesh, holeRadius);

  // Assign junctions to strips
  for (const junction of junctions) {
    for (const stripId of junction.stripIds) {
      const strip = allStrips.find(s => s.id === stripId);
      if (strip) {
        strip.junctions.push(junction);
      }
    }
  }

  // Segment strips at junctions
  for (const strip of allStrips) {
    strip.segments = segmentStripAtJunctions(strip, mesh);
  }

  return {
    strips: allStrips,
    junctions,
    families
  };
}

function calculateAverageDistance(points1: THREE.Vector3[], points2: THREE.Vector3[]): number {
  let totalDist = 0;
  let count = 0;

  const len = Math.min(points1.length, points2.length);
  for (let i = 0; i < len; i++) {
    totalDist += points1[i].distanceTo(points2[i]);
    count++;
  }

  return count > 0 ? totalDist / count : 0.1;
}

function computeCenterline(points1: THREE.Vector3[], points2: THREE.Vector3[]): THREE.Vector3[] {
  const centerline: THREE.Vector3[] = [];
  const len = Math.min(points1.length, points2.length);

  for (let i = 0; i < len; i++) {
    const center = new THREE.Vector3()
      .addVectors(points1[i], points2[i])
      .multiplyScalar(0.5);
    centerline.push(center);
  }

  return centerline;
}

function findJunctions(
  strips: Strip[],
  mesh: HalfEdgeMesh,
  holeRadius: number
): Junction[] {
  const junctions: Junction[] = [];
  let junctionId = 1;

  // Check intersections between strips of different families
  for (let i = 0; i < strips.length; i++) {
    for (let j = i + 1; j < strips.length; j++) {
      const strip1 = strips[i];
      const strip2 = strips[j];

      // Skip if same family
      if (strip1.family === strip2.family) continue;

      // Find intersection points
      const intersections = findStripIntersections(strip1, strip2, mesh);

      for (const intersection of intersections) {
        // Check if there's already a junction nearby
        const existingJunction = junctions.find(j =>
          j.position.distanceTo(intersection.point) < holeRadius * 2
        );

        if (existingJunction) {
          if (!existingJunction.stripIds.includes(strip1.id)) {
            existingJunction.stripIds.push(strip1.id);
          }
          if (!existingJunction.stripIds.includes(strip2.id)) {
            existingJunction.stripIds.push(strip2.id);
          }
        } else {
          junctions.push({
            id: junctionId++,
            position: intersection.point,
            stripIds: [strip1.id, strip2.id],
            holeRadius,
            faceIndex: intersection.faceIndex
          });
        }
      }
    }
  }

  return junctions;
}

interface Intersection {
  point: THREE.Vector3;
  faceIndex: number;
}

function findStripIntersections(
  strip1: Strip,
  strip2: Strip,
  _mesh: HalfEdgeMesh
): Intersection[] {
  const intersections: Intersection[] = [];

  // Check centerline intersections
  for (let i = 0; i < strip1.centerline.length - 1; i++) {
    const a1 = strip1.centerline[i];
    const a2 = strip1.centerline[i + 1];

    for (let j = 0; j < strip2.centerline.length - 1; j++) {
      const b1 = strip2.centerline[j];
      const b2 = strip2.centerline[j + 1];

      // Check if line segments are close enough to be considered intersecting
      const intersection = findSegmentIntersection3D(a1, a2, b1, b2);

      if (intersection) {
        const faceIndex = Math.min(
          strip1.isolines[0].faceIndices[Math.min(i, strip1.isolines[0].faceIndices.length - 1)] || 0,
          strip2.isolines[0].faceIndices[Math.min(j, strip2.isolines[0].faceIndices.length - 1)] || 0
        );

        intersections.push({
          point: intersection,
          faceIndex
        });
      }
    }
  }

  return intersections;
}

function findSegmentIntersection3D(
  a1: THREE.Vector3,
  a2: THREE.Vector3,
  b1: THREE.Vector3,
  b2: THREE.Vector3
): THREE.Vector3 | null {
  // Find closest points between two line segments in 3D
  const d1 = new THREE.Vector3().subVectors(a2, a1);
  const d2 = new THREE.Vector3().subVectors(b2, b1);
  const r = new THREE.Vector3().subVectors(a1, b1);

  const a = d1.dot(d1);
  const e = d2.dot(d2);
  const f = d2.dot(r);

  const EPSILON = 1e-6;

  let s: number, t: number;

  if (a <= EPSILON && e <= EPSILON) {
    s = t = 0;
  } else if (a <= EPSILON) {
    s = 0;
    t = Math.max(0, Math.min(1, f / e));
  } else {
    const c = d1.dot(r);
    if (e <= EPSILON) {
      t = 0;
      s = Math.max(0, Math.min(1, -c / a));
    } else {
      const b = d1.dot(d2);
      const denom = a * e - b * b;

      if (Math.abs(denom) > EPSILON) {
        s = Math.max(0, Math.min(1, (b * f - c * e) / denom));
      } else {
        s = 0;
      }

      t = (b * s + f) / e;

      if (t < 0) {
        t = 0;
        s = Math.max(0, Math.min(1, -c / a));
      } else if (t > 1) {
        t = 1;
        s = Math.max(0, Math.min(1, (b - c) / a));
      }
    }
  }

  const c1 = a1.clone().add(d1.clone().multiplyScalar(s));
  const c2 = b1.clone().add(d2.clone().multiplyScalar(t));

  const dist = c1.distanceTo(c2);

  // If segments are close enough, return midpoint
  const threshold = 0.1; // Adjust based on mesh scale
  if (dist < threshold) {
    return c1.clone().add(c2).multiplyScalar(0.5);
  }

  return null;
}

function segmentStripAtJunctions(strip: Strip, _mesh: HalfEdgeMesh): StripSegment[] {
  const segments: StripSegment[] = [];

  if (strip.junctions.length === 0) {
    // No junctions - single segment
    segments.push({
      startJunctionId: null,
      endJunctionId: null,
      points: [...strip.centerline],
      width: strip.width,
      faceIndices: strip.isolines[0].faceIndices
    });
    return segments;
  }

  // Sort junctions by position along centerline
  const sortedJunctions = sortJunctionsAlongStrip(strip);

  // Create segments between junctions
  let lastIdx = 0;

  for (let i = 0; i < sortedJunctions.length; i++) {
    const junction = sortedJunctions[i];
    const junctionIdx = findClosestPointIndex(strip.centerline, junction.position);

    if (junctionIdx > lastIdx) {
      segments.push({
        startJunctionId: i === 0 ? null : sortedJunctions[i - 1].id,
        endJunctionId: junction.id,
        points: strip.centerline.slice(lastIdx, junctionIdx + 1),
        width: strip.width,
        faceIndices: strip.isolines[0].faceIndices.slice(lastIdx, junctionIdx + 1)
      });
    }

    lastIdx = junctionIdx;
  }

  // Final segment after last junction
  if (lastIdx < strip.centerline.length - 1) {
    segments.push({
      startJunctionId: sortedJunctions[sortedJunctions.length - 1].id,
      endJunctionId: null,
      points: strip.centerline.slice(lastIdx),
      width: strip.width,
      faceIndices: strip.isolines[0].faceIndices.slice(lastIdx)
    });
  }

  return segments;
}

function sortJunctionsAlongStrip(strip: Strip): Junction[] {
  return [...strip.junctions].sort((a, b) => {
    const idxA = findClosestPointIndex(strip.centerline, a.position);
    const idxB = findClosestPointIndex(strip.centerline, b.position);
    return idxA - idxB;
  });
}

function findClosestPointIndex(points: THREE.Vector3[], target: THREE.Vector3): number {
  let minDist = Infinity;
  let minIdx = 0;

  for (let i = 0; i < points.length; i++) {
    const dist = points[i].distanceTo(target);
    if (dist < minDist) {
      minDist = dist;
      minIdx = i;
    }
  }

  return minIdx;
}

// Assign layers for over/under weaving pattern
export function assignLayers(pattern: KagomePattern): void {
  // Classic Kagome: each family goes over one family and under another
  // Family A (0): over B, under C
  // Family B (1): over C, under A
  // Family C (2): over A, under B

  for (const junction of pattern.junctions) {
    const stripFamilies = junction.stripIds.map(id => {
      const strip = pattern.strips.find(s => s.id === id);
      return strip ? strip.family : -1;
    }).filter(f => f >= 0);

    if (stripFamilies.length >= 2) {
      // Determine layer order at this junction
      for (const stripId of junction.stripIds) {
        const strip = pattern.strips.find(s => s.id === stripId);
        if (!strip) continue;

        const otherFamilies = stripFamilies.filter(f => f !== strip.family);

        // Assign layer based on weaving pattern
        let layer = 1; // Middle by default

        for (const otherFamily of otherFamilies) {
          if ((strip.family + 1) % 3 === otherFamily) {
            // This strip goes over
            layer = 2;
          } else if ((strip.family + 2) % 3 === otherFamily) {
            // This strip goes under
            layer = 0;
          }
        }

        strip.layer = layer;
      }
    }
  }
}

// Generate hole geometry at junctions
export function generateJunctionHoles(
  junctions: Junction[],
  radius: number
): { centers: THREE.Vector3[]; radii: number[] } {
  return {
    centers: junctions.map(j => j.position.clone()),
    radii: junctions.map(() => radius)
  };
}
