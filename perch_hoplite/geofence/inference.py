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

"""Inference for species geofencing."""

from etils import epath
import pandas as pd
from perch_hoplite import path_utils

import s2geometry as s2

INDEX_PATH = path_utils.get_absolute_path("geofence/species_index.parquet")


class GeofenceInference:
  """Loads a geofence index and returns species for a given lat/lon."""

  def __init__(self, index_path: epath.Path | str | None = None):
    """Initializes the GeofenceInference.

    Args:
      index_path: Path to the Parquet file containing the geofence index.
    """
    if index_path is None:
      index_path = INDEX_PATH
    self.index_path = epath.Path(index_path)
    self._species_unions: list[tuple[str, s2.S2CellUnion]] = []
    with self.index_path.open("rb") as f:
      df = pd.read_parquet(f)
    for _, row in df.iterrows():
      decoder = s2.Decoder(row["cell_union"])
      cell_union = s2.S2CellUnion([])
      if cell_union.decode(decoder):
        self._species_unions.append((row["species_id"], cell_union))
      else:
        print(f"Failed to decode S2CellUnion for {row['species_id']}")

  def get_species_for_lat_lon(self, lat: float, lon: float) -> list[str]:
    """Returns a list of species whose ranges contain the given lat/lon.

    Args:
      lat: Latitude of the point.
      lon: Longitude of the point.

    Returns:
      A list of species names.
    """
    point = s2.S2LatLng.from_degrees(lat, lon).to_point()
    result = []
    for species_id, cell_union in self._species_unions:
      if cell_union.contains_point(point):
        result.append(species_id)
    return result
