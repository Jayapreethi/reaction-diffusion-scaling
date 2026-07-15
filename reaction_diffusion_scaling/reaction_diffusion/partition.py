"""
Strip-based domain decomposition with ghost-cell exchange.

Partitions the 2D domain into horizontal strips (splitting along rows only).
Each partition maintains ghost rows for communication with neighbors.
"""

from typing import List, Tuple
import torch


class PartitionedDomain:
    """
    Represents a single horizontal strip partition with ghost rows.
    
    Parameters:
        global_grid_size (int): Total number of rows in the global domain.
        num_rows (int): Number of locally owned rows (excluding ghosts).
        start_row (int): Starting row index in the global domain.
        partition_id (int): Identifier for this partition.
        num_partitions (int): Total number of partitions.
        device (torch.device): Device for tensors.
        dtype (torch.dtype): Data type for tensors.
    """
    
    def __init__(
        self,
        global_grid_size: int,
        num_rows: int,
        start_row: int,
        partition_id: int,
        num_partitions: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.global_grid_size = global_grid_size
        self.num_rows = num_rows  # Locally owned rows
        self.start_row = start_row
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        self.device = device
        self.dtype = dtype
        
        # Determine if we have neighbors
        self.has_top_neighbor = (partition_id > 0)
        self.has_bottom_neighbor = (partition_id < num_partitions - 1)
        
        # Total rows including ghosts
        self.total_rows = num_rows
        if self.has_top_neighbor:
            self.total_rows += 1
        if self.has_bottom_neighbor:
            self.total_rows += 1
        
        # Ghost row offsets (in local coordinates)
        # If we have a top neighbor, the ghost row is at index 0
        # Owned rows start at index 1 (if has_top_neighbor) or 0 (if not)
        # If we have a bottom neighbor, ghost row is at the end
        
        # Local indices for owned rows
        if self.has_top_neighbor:
            self.owned_start = 1
        else:
            self.owned_start = 0
        
        self.owned_end = self.owned_start + self.num_rows
    
    def create_field(self) -> torch.Tensor:
        """Create an empty field for this partition."""
        return torch.zeros(
            (self.total_rows, self.global_grid_size),
            dtype=self.dtype,
            device=self.device,
        )
    
    def set_owned_rows(self, field: torch.Tensor, global_field: torch.Tensor):
        """
        Set the owned rows from a global field.
        
        Parameters:
            field (torch.Tensor): Local partitioned field (including ghosts).
            global_field (torch.Tensor): Global field.
        """
        field[self.owned_start:self.owned_end, :] = global_field[
            self.start_row:self.start_row + self.num_rows, :
        ]
    
    def get_owned_rows(self, field: torch.Tensor) -> torch.Tensor:
        """
        Extract owned rows (excluding ghosts).
        
        Parameters:
            field (torch.Tensor): Local partitioned field (including ghosts).
        
        Returns:
            owned (torch.Tensor): Just the owned rows.
        """
        return field[self.owned_start:self.owned_end, :]
    
    def get_top_boundary_row(self, field: torch.Tensor) -> torch.Tensor:
        """
        Get the top boundary row to send to the top neighbor.
        This is the topmost owned row.
        
        Parameters:
            field (torch.Tensor): Local partitioned field (including ghosts).
        
        Returns:
            row (torch.Tensor): Top boundary row.
        """
        return field[self.owned_start, :].clone()
    
    def get_bottom_boundary_row(self, field: torch.Tensor) -> torch.Tensor:
        """
        Get the bottom boundary row to send to the bottom neighbor.
        This is the bottommost owned row.
        
        Parameters:
            field (torch.Tensor): Local partitioned field (including ghosts).
        
        Returns:
            row (torch.Tensor): Bottom boundary row.
        """
        return field[self.owned_end - 1, :].clone()
    
    def set_top_ghost_row(self, field: torch.Tensor, ghost_row: torch.Tensor):
        """
        Set the top ghost row from the top neighbor.
        
        Parameters:
            field (torch.Tensor): Local partitioned field.
            ghost_row (torch.Tensor): Ghost row from top neighbor.
        """
        if self.has_top_neighbor:
            field[0, :] = ghost_row
    
    def set_bottom_ghost_row(self, field: torch.Tensor, ghost_row: torch.Tensor):
        """
        Set the bottom ghost row from the bottom neighbor.
        
        Parameters:
            field (torch.Tensor): Local partitioned field.
            ghost_row (torch.Tensor): Ghost row from bottom neighbor.
        """
        if self.has_bottom_neighbor:
            field[-1, :] = ghost_row


class StripDecomposer:
    """
    Manages strip-based domain decomposition.
    
    Parameters:
        global_grid_size (int): Grid size (square domain).
        num_partitions (int): Number of horizontal strips.
        device (torch.device): Device for tensors.
        dtype (torch.dtype): Data type for tensors.
    """
    
    def __init__(
        self,
        global_grid_size: int,
        num_partitions: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.global_grid_size = global_grid_size
        self.num_partitions = num_partitions
        self.device = device
        self.dtype = dtype
        
        # Compute row distribution (uneven decomposition handled)
        self.rows_per_partition, self.row_starts = self._compute_distribution()
        
        # Create partition objects
        self.partitions: List[PartitionedDomain] = []
        for pid in range(num_partitions):
            partition = PartitionedDomain(
                global_grid_size=global_grid_size,
                num_rows=self.rows_per_partition[pid],
                start_row=self.row_starts[pid],
                partition_id=pid,
                num_partitions=num_partitions,
                device=device,
                dtype=dtype,
            )
            self.partitions.append(partition)
    
    def _compute_distribution(self) -> Tuple[List[int], List[int]]:
        """
        Compute row distribution across partitions.
        Handles uneven decomposition by distributing remainder rows.
        
        Returns:
            rows_per_partition (List[int]): Number of owned rows per partition.
            row_starts (List[int]): Starting global row index for each partition.
        """
        base_rows = self.global_grid_size // self.num_partitions
        remainder = self.global_grid_size % self.num_partitions
        
        rows_per_partition = []
        row_starts = []
        current_start = 0
        
        for pid in range(self.num_partitions):
            # First 'remainder' partitions get an extra row
            num_rows = base_rows + (1 if pid < remainder else 0)
            rows_per_partition.append(num_rows)
            row_starts.append(current_start)
            current_start += num_rows
        
        return rows_per_partition, row_starts
    
    def decompose_global_field(self, global_field: torch.Tensor) -> List[torch.Tensor]:
        """
        Decompose a global field into partitioned fields.
        
        Parameters:
            global_field (torch.Tensor): Global field of shape (global_grid_size, global_grid_size).
        
        Returns:
            partitioned_fields (List[torch.Tensor]): List of local fields (with ghost rows).
        """
        partitioned_fields = []
        
        for partition in self.partitions:
            local_field = partition.create_field()
            partition.set_owned_rows(local_field, global_field)
            partitioned_fields.append(local_field)
        
        return partitioned_fields
    
    def exchange_ghosts(self, partitioned_fields: List[torch.Tensor]):
        """
        Exchange ghost rows between neighboring partitions.
        Updates ghost rows in place using the current owned rows of neighbors.
        
        Parameters:
            partitioned_fields (List[torch.Tensor]): Local fields for all partitions.
        """
        for pid in range(self.num_partitions):
            partition = self.partitions[pid]
            field = partitioned_fields[pid]
            
            # Exchange with top neighbor
            if partition.has_top_neighbor:
                top_neighbor_field = partitioned_fields[pid - 1]
                top_ghost = self.partitions[pid - 1].get_bottom_boundary_row(top_neighbor_field)
                partition.set_top_ghost_row(field, top_ghost)
            
            # Exchange with bottom neighbor
            if partition.has_bottom_neighbor:
                bottom_neighbor_field = partitioned_fields[pid + 1]
                bottom_ghost = self.partitions[pid + 1].get_top_boundary_row(bottom_neighbor_field)
                partition.set_bottom_ghost_row(field, bottom_ghost)
    
    def reassemble_global_field(self, partitioned_fields: List[torch.Tensor]) -> torch.Tensor:
        """
        Reassemble local partitioned fields into a global field.
        
        Parameters:
            partitioned_fields (List[torch.Tensor]): Local fields for all partitions.
        
        Returns:
            global_field (torch.Tensor): Reassembled global field.
        """
        global_field = torch.zeros(
            (self.global_grid_size, self.global_grid_size),
            dtype=self.dtype,
            device=self.device,
        )
        
        for pid, partition in enumerate(self.partitions):
            field = partitioned_fields[pid]
            owned_rows = partition.get_owned_rows(field)
            start_row = partition.start_row
            num_rows = partition.num_rows
            
            global_field[start_row:start_row + num_rows, :] = owned_rows
        
        return global_field
