"""Assembly graph: unique IDs, a single rooted tree, and no connection cycles."""

from __future__ import annotations

from typing import Any

from talongym.robot.contract import AssemblyError


def index_instances(assembly: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in assembly.get("instances") or []:
        ident = str(row.get("id") or "")
        if not ident:
            raise AssemblyError("assembly instance is missing id")
        if ident in rows:
            raise AssemblyError(f"duplicate assembly instance id {ident}")
        if ident == "chassis":
            raise AssemblyError("instance id chassis is reserved for the compiled physical root")
        rows[ident] = row
    if not rows:
        raise AssemblyError("assembly must contain at least one instance")
    return rows


def index_connections(assembly: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in assembly.get("connections") or []:
        ident = str(row.get("id") or "")
        if not ident:
            raise AssemblyError("assembly connection is missing id")
        if ident in rows:
            raise AssemblyError(f"duplicate assembly connection id {ident}")
        rows[ident] = row
    return rows


def root_instance_id(assembly: dict[str, Any], instances: dict[str, dict[str, Any]]) -> str:
    wanted = str(assembly.get("rootInstanceId") or "")
    if wanted:
        if wanted not in instances:
            raise AssemblyError(f"rootInstanceId {wanted} is not an assembly instance")
        return wanted
    if len(instances) == 1:
        return next(iter(instances))
    raise AssemblyError("assembly with multiple instances requires rootInstanceId")


def parent_map(
    connections: dict[str, dict[str, Any]],
    instances: dict[str, dict[str, Any]],
) -> dict[str, str]:
    parents: dict[str, str] = {}
    for ident, connection in connections.items():
        parent_ref = connection.get("parent") or {}
        child_ref = connection.get("child") or {}
        parent_id = str(parent_ref.get("instanceId") or "")
        child_id = str(child_ref.get("instanceId") or "")
        if parent_id not in instances:
            raise AssemblyError(f"connection {ident} parent instance {parent_id} is missing")
        if child_id not in instances:
            raise AssemblyError(f"connection {ident} child instance {child_id} is missing")
        if parent_id == child_id:
            raise AssemblyError(f"connection {ident} cannot parent an instance to itself")
        if child_id in parents:
            raise AssemblyError(f"instance {child_id} has multiple parents")
        parents[child_id] = parent_id
    return parents


def undirected_adjacency(
    connections: dict[str, dict[str, Any]] | list[dict[str, Any]],
) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    rows = connections.values() if isinstance(connections, dict) else connections
    adjacency: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for connection in rows:
        parent_id = str(connection["parent"]["instanceId"])
        child_id = str(connection["child"]["instanceId"])
        adjacency.setdefault(parent_id, []).append((child_id, connection))
        adjacency.setdefault(child_id, []).append((parent_id, connection))
    return adjacency


def subtree_ids(root_id: str, by_child: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    children: dict[str, list[str]] = {}
    for child_id, connection in by_child.items():
        parent_id = str(connection["parent"]["instanceId"])
        children.setdefault(parent_id, []).append(str(child_id))
    ordered = [root_id]
    stack = [root_id]
    while stack:
        current = stack.pop()
        for child in children.get(current, []):
            ordered.append(child)
            stack.append(child)
    return tuple(ordered)


def validate_assembly_graph(assembly: dict[str, Any]) -> tuple[str, tuple[str, ...], dict[str, dict[str, Any]]]:
    """Return root id, BFS instance order, and connections keyed by child instance."""
    instances = index_instances(assembly)
    connections = index_connections(assembly)
    root = root_instance_id(assembly, instances)
    parents = parent_map(connections, instances)
    if root in parents:
        raise AssemblyError(f"root instance {root} cannot be a connection child")
    by_child = {
        str(row["child"]["instanceId"]): row
        for row in connections.values()
    }
    visiting: set[str] = set()
    ordered: list[str] = []

    def visit(ident: str) -> None:
        if ident in ordered:
            return
        if ident in visiting:
            raise AssemblyError(f"assembly connection cycle at {ident}")
        visiting.add(ident)
        parent = parents.get(ident)
        if parent is not None:
            visit(parent)
        visiting.remove(ident)
        ordered.append(ident)

    for ident in instances:
        visit(ident)
    disconnected = sorted(ident for ident in instances if ident != root and ident not in parents)
    if disconnected:
        raise AssemblyError(f"disconnected assembly instances: {disconnected}")
    for ident in instances:
        if ident == root:
            continue
        current = ident
        seen: set[str] = set()
        while current != root:
            if current in seen:
                raise AssemblyError(f"assembly connection cycle at {current}")
            seen.add(current)
            if current not in parents:
                raise AssemblyError(f"disconnected assembly instances: {[ident]}")
            current = parents[current]
    return root, tuple(ordered), by_child
