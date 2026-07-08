# coding=utf-8
# Copyright 2026 The Perch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library for creating species geofencing tables."""

import concurrent.futures
import time

import geopandas as gpd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

import s2geometry as s2

# Skip species with errors in the geometry.
SKIP_SPECIES = [
    "Columba livia",
    "Pygoscelis adeliae",
    "Aptenodytes forsteri",
    "Calidris canutus",
    "Calidris alpina",
    "Oceanites oceanicus",
    "Pagodroma nivea",
    "Thalassoica antarctica",
    "Larus hyperboreus",
    "Larus fuscus",
    "Sterna paradisaea",
    "Stercorarius parasiticus",
    "Rissa tridactyla",
    "Xema sabini",
    "Gavia stellata",
    "Falco peregrinus",
    "Pluvialis squatarola",
    "Mergus serrator",
    "Branta bernicla",
    "Clangula hyemalis",
    "Asio flammeus",
    "Phalaropus fulicarius",
    "Calcarius lapponicus",
    "Stercorarius maccormicki",
]


def _shapely_point_to_s2_point(p):
  return s2.S2LatLng.from_degrees(p[1], p[0]).to_point()


def _shapely_ring_to_s2_loop(ring):
  """Converts a shapely LinearRing to an S2Loop."""
  points = [_shapely_point_to_s2_point(p) for p in ring.coords[:-1]]
  if len(points) < 3:
    return None
  loop = s2.S2Loop(points)
  # s2 loops must be oriented CCW.
  # loop.normalize() ensures CCW orientation.
  loop.normalize()
  return loop


def _shapely_polygon_to_s2_polygon(poly):
  """Converts a shapely Polygon to an S2Polygon."""
  s2poly = s2.S2Polygon()
  loops = []
  try:
    exterior = _shapely_ring_to_s2_loop(poly.exterior)
  except Exception:  # pylint: disable=broad-except
    return None
  if exterior is None:
    return None
  loops.append(exterior)
  for interior in poly.interiors:
    try:
      interior_loop = _shapely_ring_to_s2_loop(interior)
      if interior_loop is not None:
        loops.append(interior_loop)
    except Exception:  # pylint: disable=broad-except
      # Skip invalid interior loops.
      continue
  s2poly.init_nested(loops)
  return s2poly


def get_s2_covering(
    geometry_obj,
    min_level: int = 2,
    max_level: int = 9,
    max_cells: int = 512,
    simplify_tolerance: float = 0.005,
) -> s2.S2CellUnion:
  """Returns a list of (id_start, id_end) tuples for any Shapely geometry."""
  # 0. Fix invalid geometries.
  geometry_obj = geometry_obj.buffer(0)
  if geometry_obj.is_empty:
    return s2.S2CellUnion([])
  # 1. Simplify to skip micro-details (biological ranges are rarely cm-precise)
  simplified_geom = geometry_obj.simplify(
      simplify_tolerance, preserve_topology=True
  )
  geom = simplified_geom
  if simplified_geom.is_empty and not geometry_obj.is_empty:
    geom = geometry_obj

  options = s2.S2RegionCoverer.Options()
  options.min_level = min_level
  options.max_level = max_level
  options.max_cells = max_cells
  coverer = s2.S2RegionCoverer(options)

  if geom.geom_type == "Polygon":
    polygons = [geom]
  elif geom.geom_type == "MultiPolygon":
    polygons = list(geom.geoms)
  else:
    print(f"Unsupported geometry type: {geom.geom_type}")
    return s2.S2CellUnion([])

  all_cell_ids = set()
  for poly in polygons:
    if poly.is_empty:
      continue
    s2poly = _shapely_polygon_to_s2_polygon(poly)
    if s2poly is None or s2poly.is_empty:
      continue
    covering = coverer.cover(s2poly)
    all_cell_ids.update(covering)
  return s2.S2CellUnion(list(all_cell_ids))


def build_parquet_index(
    gdf: gpd.GeoDataFrame,
    output_file: str | None = "/tmp/species_index.parquet",
    min_level: int = 2,
    max_level: int = 9,
    max_cells: int = 512,
    simplify_tolerance: float = 0.005,
    max_workers: int = 16,
) -> pa.Table:
  """Processes the GeoDataFrame and saves it to a compressed Parquet file.

  Args:
    gdf: GeoDataFrame with geometry column containing species ranges. Expected
      to be in EPSG:4326.
    output_file: If not None, path to write the resulting parquet index.
    min_level: Minimum S2 level for cells in covering.
    max_level: Maximum S2 level for cells in covering.
    max_cells: Maximum number of cells in covering.
    simplify_tolerance: Tolerance for geometry simplification.
    max_workers: Number of threads for concurrent processing.

  Returns:
    Arrow table containing species_id and encoded S2CellUnion.
  """
  species_unions: list[tuple[str, s2.S2CellUnion]] = []

  def process_row(args):
    """Get s2 covering for one row."""
    idx, row = args
    name = row.get("name", f"sp_{idx}")
    if name in SKIP_SPECIES:
      print(f"Skipping {name}...")
      return name, s2.S2CellUnion([])
    try:
      if row["geometry"].is_empty:
        print(f"Skipping {name} due to empty geometry...")
        return name, s2.S2CellUnion([])
      start_time = time.time()
      cell_union = get_s2_covering(
          row["geometry"],
          min_level=min_level,
          max_level=max_level,
          max_cells=max_cells,
          simplify_tolerance=simplify_tolerance,
      )
      elapsed = time.time() - start_time
      if not cell_union.cell_ids and not row["geometry"].is_empty:
        print(f"!!! {name} resulted in 0 cells !!!")
      print(
          f"completed {idx} {name} with {len(cell_union.cell_ids)} cells in"
          f" {elapsed:.2f}s"
      )
      return name, cell_union
    except Exception as e:  # pylint: disable=broad-except
      print(f"\nError processing {name}: {e}")
      return name, s2.S2CellUnion([])

  with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
    results = tqdm.tqdm(
        executor.map(process_row, gdf.iterrows()),
        total=len(gdf),
        desc="Processing species",
    )
    for name, cell_union in results:
      species_unions.append((name, cell_union))

  all_names = []
  all_encoded_unions = []
  for name, cell_union in species_unions:
    if not cell_union.cell_ids:
      continue
    encoder = s2.Encoder()
    cell_union.encode(encoder)
    all_names.append(name)
    all_encoded_unions.append(encoder.buffer())

  table = pa.table({
      "species_id": all_names,
      "cell_union": pa.array(all_encoded_unions, type=pa.binary()),
  })

  if output_file:
    pq.write_table(table, output_file, compression="snappy")
  return table
