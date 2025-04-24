import argparse
import json
from pathlib import Path
from typing import List
import csv

import numpy as np
import torch
import open3d as o3d
import enum


class Layers(enum.Enum):
    SEGMENTS = 1
    OBJECTS = 2
    PLACES = 3
    ROOMS = 4
    BUILDINGS = 5
    AGENTS = 0


type_2_layer = {
    "ObjectNodeAttributes": Layers.OBJECTS,
    "PlaceNodeAttributes": Layers.PLACES,
    "RoomNodeAttributes": Layers.ROOMS,
}


class Edge:
    def __init__(self, edge_id, source_id, target_id):
        self.edge_id = edge_id
        self.source_id = source_id
        self.target_id = target_id
        self.embedding: torch.Tensor = None

    def to_dict(self):
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "embedding": self.embedding.tolist() if self.embedding is not None else "",
        }


class Object:
    """
    Class to represent an object in a room.
    :param object_id: Unique identifier for the object
    :param room_id: Identifier of the room this object belongs to
    :param name: Name of the object (e.g., "Chair", "Table")
    """

    def __init__(self, object_id, room_id=None, name=None):
        self.object_id = object_id  # Unique identifier for the object
        self.vertices: np.ndarray = (
            None  # Coordinates of the object in the point cloud 8 vertices
        )
        self.embedding: torch.Tensor = None  # CLIP Embedding of the object
        self.pcd = None  # Point cloud of the object
        self.room_id = room_id  # Identifier of the room this object belongs to
        self.name = name  # Name of the object (e.g., "Chair", "Table")
        self.gt_name = None

    def save(self, path: Path):
        """
        Save the object in folder as ply for the point cloud
        and json for the metadata
        """
        # save the point cloud
        o3d.io.write_point_cloud(str(path / f"{self.object_id}.ply"), self.pcd)
        # save the metadata
        metadata = {
            "object_id": self.object_id,
            "vertices": np.array(self.vertices).tolist(),
            "room_id": self.room_id,
            "name": self.name,
            "embedding": self.embedding.tolist() if self.embedding is not None else "",
        }
        with (path / f"{self.object_id}.json").open("w") as outfile:
            json.dump(metadata, outfile, indent=2)


class Place:
    def __init__(self, place_id, name=None):
        self.place_id = place_id
        self.name = name
        self.object_ids = []


class Room:
    """
    Class to represent a room in a building.
    :param room_id: Unique identifier for the room
    :param floor_id: Identifier of the floor this room belongs to
    :param name: Name of the room (e.g., "Living Room", "Bedroom")
    """

    def __init__(self, room_id, floor_id=0, name=None):
        self.room_id = room_id  # Unique identifier for the room
        self.name = name  # Name of the room (e.g., "Living Room", "Bedroom")
        self.category = None  # placeholder for a GT category
        self.floor_id = floor_id  # Identifier of the floor this room belongs to
        self.objects: List[Object] = []  # List of objects inside the room
        self.vertices: np.ndarray = np.asarray(
            []
        )  # indices of the room in the point cloud 8 vertices
        self.embeddings: List[
            torch.Tensor
        ] = []  # List of tensors of embeddings of the room
        self.pcd = None  # Point cloud of the room
        self.room_height = None  # Height of the room
        self.room_zero_level = None  # Zero level of the room
        self.represent_images = []  # 5 images that represent the appearance of the room
        self.object_counter = 0
        self.places_ids = []

    def save(self, path: Path):
        """
        Save the room in folder as ply for the point cloud
        and json for the metadata
        """
        # save the metadata
        metadata = {
            "room_id": self.room_id,
            "name": self.name,
            "floor_id": self.floor_id,
            "objects": [obj.object_id for obj in self.objects],
            "vertices": self.vertices.tolist(),
            "room_height": self.room_height,
            "room_zero_level": self.room_zero_level,
            "embeddings": [i.tolist() for i in self.embeddings],
            "represent_images": self.represent_images,
        }
        with (path / f"{self.room_id}.json").open("w") as outfile:
            json.dump(metadata, outfile, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("colormap_path", type=str)
    return parser.parse_args()


def parse_colormap(colormap_path: Path):
    id_to_rgb = {}
    with colormap_path.open(mode="r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            obj_id = int(row["id"])
            rgb = np.array(
                [int(row["red"]), int(row["green"]), int(row["blue"])], dtype=np.uint8
            )

            if obj_id not in id_to_rgb:
                id_to_rgb[obj_id] = rgb

    return id_to_rgb


def main(input_file: Path, output_file: Path, colormap_path: Path):
    with input_file.open("r") as f:
        hydra_data = json.load(f)

    colormap = parse_colormap(colormap_path)
    # Set default to gray
    colormap[-1] = np.array([128, 128, 128], dtype=np.uint8)

    # Parse mesh data
    points = np.asarray(hydra_data["mesh"]["points"])
    faces = np.asarray(hydra_data["mesh"]["faces"])
    colors = None
    labels = None
    timestamps = None
    if hydra_data["mesh"]["has_colors"]:
        # RGBA
        colors = np.asarray(
            [
                [color["r"], color["g"], color["b"], color["a"]]
                for color in hydra_data["mesh"]["colors"]
            ]
        )
    if hydra_data["mesh"]["has_labels"]:
        labels = np.asarray(hydra_data["mesh"]["labels"])
        labels[labels == 4294967295] = -1
    if hydra_data["mesh"]["has_timestamps"]:
        timestamps = np.asarray(hydra_data["mesh"]["stamps"])

    # Parse objects, rooms, places
    places_dict = dict()
    rooms_dict = dict()
    objects_dict = dict()
    for node_data in hydra_data["nodes"]:
        node_type = node_data["attributes"]["type"]
        if node_type == "ObjectNodeAttributes":
            node = Object(node_data["id"])
            node.name = node_data["attributes"]["name"]
            bb_dims = np.asarray(node_data["attributes"]["bounding_box"]["dimensions"])
            bb_center = np.asarray(
                node_data["attributes"]["bounding_box"]["world_P_center"]
            )
            node.vertices = np.array(
                [
                    bb_center + bb_dims / 2 * np.array([1, 1, 1]),
                    bb_center + bb_dims / 2 * np.array([1, 1, -1]),
                    bb_center + bb_dims / 2 * np.array([1, -1, 1]),
                    bb_center + bb_dims / 2 * np.array([1, -1, -1]),
                    bb_center + bb_dims / 2 * np.array([-1, 1, 1]),
                    bb_center + bb_dims / 2 * np.array([-1, 1, -1]),
                    bb_center + bb_dims / 2 * np.array([-1, -1, 1]),
                    bb_center + bb_dims / 2 * np.array([-1, -1, -1]),
                ]
            )
            node.embedding = (
                torch.tensor(node_data["attributes"]["semantic_feature"])
                if node_data["attributes"]["semantic_feature"] is not None
                else None
            )
            node.pcd = o3d.geometry.PointCloud()
            node.pcd.points = o3d.utility.Vector3dVector(
                points[node_data["attributes"]["mesh_connections"]]
            )
            node.pcd.colors = o3d.utility.Vector3dVector(
                colors[node_data["attributes"]["mesh_connections"]][:, :3].astype(float)
                / 255.0
            )
            objects_dict[node.object_id] = node
        elif node_type == "PlaceNodeAttributes":
            node = Place(node_data["id"])
            places_dict[node.place_id] = node
        elif node_type == "RoomNodeAttributes":
            node = Room(node_data["id"])
            node.name = node_data["attributes"]["name"]
            node.embeddings = [
                torch.tensor(embedding)
                for embedding in node_data["attributes"]["feature_vectors"]
            ]
            rooms_dict[node.room_id] = node
        else:
            continue

    # Parse edges
    object_edges = dict()
    interlayer_edges = dict()
    for i, edge in enumerate(hydra_data["edges"]):
        source_id = edge["source"]
        target_id = edge["target"]
        if source_id in objects_dict and target_id in objects_dict:
            edge1 = Edge(i, source_id, target_id)
            edge1.embedding = torch.tensor(
                edge["info"]["relationship_source_target.feature"]["data"]
            ).reshape(
                edge["info"]["relationship_source_target.feature"]["rows"],
                edge["info"]["relationship_source_target.feature"]["cols"],
            )
            edge2 = Edge(i, target_id, source_id)
            edge2.embedding = torch.tensor(
                edge["info"]["relationship_target_source.feature"]["data"]
            ).reshape(
                edge["info"]["relationship_target_source.feature"]["rows"],
                edge["info"]["relationship_target_source.feature"]["cols"],
            )
            object_edges[f"{source_id},{target_id}"] = edge1.to_dict()
            object_edges[f"{target_id},{source_id}"] = edge2.to_dict()
        else:
            interlayer_edges[i] = Edge(i, source_id, target_id).to_dict()

            if source_id in objects_dict or target_id in objects_dict:
                if source_id in objects_dict:
                    places_dict[target_id].object_ids.append(source_id)
                else:
                    places_dict[source_id].object_ids.append(target_id)
            elif source_id in rooms_dict and target_id in places_dict:
                rooms_dict[source_id].places_ids.append(target_id)
            elif source_id in places_dict and target_id in rooms_dict:
                rooms_dict[target_id].places_ids.append(source_id)

    # Set objects room
    for room in rooms_dict.values():
        for place_id in room.places_ids:
            for object_id in places_dict[place_id].object_ids:
                objects_dict[object_id].room_id = room.room_id
    # Set room objects
    for object in objects_dict.values():
        if object.room_id is not None:
            rooms_dict[object.room_id].objects.append(object)

    # Save objects
    output_file.mkdir(exist_ok=True, parents=True)
    (output_file / "objects").mkdir(exist_ok=True)
    for object in objects_dict.values():
        object.save(output_file / "objects")

    # Save rooms
    (output_file / "rooms").mkdir(exist_ok=True)
    for room in rooms_dict.values():
        room.save(output_file / "rooms")

    # Save the edges
    with (output_file / "object_edges.json").open("w") as outfile:
        json.dump(object_edges, outfile, indent=2)
    with (output_file / "interlayer_edges.json").open("w") as outfile:
        json.dump(interlayer_edges, outfile, indent=2)

    # Save the colored mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors[:, :3].astype(float) / 255.0)
    o3d.io.write_triangle_mesh(str(output_file / "rgb_mesh.ply"), mesh)

    # Save the colored point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3].astype(float) / 255.0)
    o3d.io.write_point_cloud(str(output_file / "rgb_pcd.ply"), pcd)

    # Save the labels
    np.savetxt(output_file / "labels.txt", labels)

    # Save the timestamps
    np.savetxt(output_file / "timestamps.txt", timestamps)

    # Save the mesh colored by labels
    if labels is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            [colormap[label].astype(float) / 255.0 for label in labels]
        )
        o3d.io.write_triangle_mesh(str(output_file / "label_mesh.ply"), mesh)
        pcd.colors = o3d.utility.Vector3dVector(
            [colormap[label].astype(float) / 255.0 for label in labels]
        )
        o3d.io.write_point_cloud(str(output_file / "label_pcd.ply"), pcd)


if __name__ == "__main__":
    args = parse_args()
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    colormap_path = Path(args.colormap_path)
    main(input_file, output_file, colormap_path)
