"""
Unit tests for the partition and domain decomposition module.
"""

import torch
import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from reaction_diffusion.partition import PartitionedDomain, StripDecomposer


class TestPartitionedDomain:
    """Tests for PartitionedDomain."""
    
    def test_partition_creation(self):
        """Test creating a single partition."""
        partition = PartitionedDomain(
            global_grid_size=128,
            num_rows=64,
            start_row=0,
            partition_id=0,
            num_partitions=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        
        assert partition.num_rows == 64
        assert partition.start_row == 0
        assert partition.partition_id == 0
        assert partition.num_partitions == 2
    
    def test_partition_with_neighbors(self):
        """Test partition neighbor detection."""
        # First partition (has bottom neighbor)
        p0 = PartitionedDomain(
            global_grid_size=128,
            num_rows=64,
            start_row=0,
            partition_id=0,
            num_partitions=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert p0.has_top_neighbor is False
        assert p0.has_bottom_neighbor is True
        
        # Middle partition (has both neighbors)
        p1 = PartitionedDomain(
            global_grid_size=256,
            num_rows=64,
            start_row=64,
            partition_id=1,
            num_partitions=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert p1.has_top_neighbor is True
        assert p1.has_bottom_neighbor is True
        
        # Last partition (has top neighbor)
        p2 = PartitionedDomain(
            global_grid_size=256,
            num_rows=64,
            start_row=192,
            partition_id=2,
            num_partitions=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert p2.has_top_neighbor is True
        assert p2.has_bottom_neighbor is False
    
    def test_field_creation(self):
        """Test creating a field for a partition."""
        partition = PartitionedDomain(
            global_grid_size=128,
            num_rows=64,
            start_row=0,
            partition_id=0,
            num_partitions=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        
        field = partition.create_field()
        
        # Should have proper shape (64 rows + 1 ghost row for bottom neighbor)
        assert field.shape[0] == 65
        assert field.shape[1] == 128
    
    def test_ghost_row_management(self):
        """Test setting and getting ghost rows."""
        partition = PartitionedDomain(
            global_grid_size=128,
            num_rows=64,
            start_row=0,
            partition_id=0,
            num_partitions=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        
        field = partition.create_field()
        
        # Set bottom boundary row
        boundary_row = torch.ones(128)
        field[partition.owned_end - 1, :] = boundary_row
        
        # Get bottom boundary row
        retrieved = partition.get_bottom_boundary_row(field)
        assert torch.allclose(retrieved, boundary_row)


class TestStripDecomposer:
    """Tests for StripDecomposer."""
    
    def test_decomposer_creation(self):
        """Test creating a decomposer."""
        decomposer = StripDecomposer(
            global_grid_size=128,
            num_partitions=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        
        assert decomposer.global_grid_size == 128
        assert decomposer.num_partitions == 2
        assert len(decomposer.partitions) == 2
    
    def test_even_decomposition(self):
        """Test decomposition with evenly divisible grid."""
        decomposer = StripDecomposer(
            global_grid_size=128,
            num_partitions=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        
        # Each partition should have 32 rows
        for partition in decomposer.partitions:
            assert partition.num_rows == 32
    
    def test_uneven_decomposition(self):
        """Test decomposition with unevenly divisible grid."""
        decomposer = StripDecomposer(
            global_grid_size=100,
            num_partitions=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        
        # 100 / 3 = 33, remainder 1
        # First partition gets extra row: 34 rows
        # Others: 33 rows
        assert decomposer.partitions[0].num_rows == 34
        assert decomposer.partitions[1].num_rows == 33
        assert decomposer.partitions[2].num_rows == 33
        
        # Total should still be 100
        total_rows = sum(p.num_rows for p in decomposer.partitions)
        assert total_rows == 100
    
    def test_decompose_global_field(self):
        """Test decomposing a global field."""
        decomposer = StripDecomposer(
            global_grid_size=64,
            num_partitions=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        
        # Create a global field with distinct values by row
        global_field = torch.arange(64, dtype=torch.float32).unsqueeze(1).expand(-1, 64)
        
        partitioned_fields = decomposer.decompose_global_field(global_field)
        
        # Check that partitions have correct values in owned rows
        for pid, partition in enumerate(decomposer.partitions):
            field = partitioned_fields[pid]
            owned_rows = partition.get_owned_rows(field)
            expected_rows = global_field[
                partition.start_row:partition.start_row + partition.num_rows, :
            ]
            assert torch.allclose(owned_rows, expected_rows)
    
    def test_reassemble_global_field(self):
        """Test reassembling a global field from partitions."""
        decomposer = StripDecomposer(
            global_grid_size=64,
            num_partitions=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        
        # Create a global field
        global_field = torch.randn(64, 64)
        
        # Decompose and reassemble
        partitioned_fields = decomposer.decompose_global_field(global_field)
        reassembled = decomposer.reassemble_global_field(partitioned_fields)
        
        # Should recover the original (since no modifications were made)
        assert torch.allclose(reassembled, global_field)
    
    def test_ghost_exchange_simple(self):
        """Test ghost exchange between two partitions."""
        decomposer = StripDecomposer(
            global_grid_size=64,
            num_partitions=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        
        # Create a field with distinct values by row
        global_field = torch.arange(64, dtype=torch.float32).unsqueeze(1).expand(-1, 64) / 64.0
        
        partitioned_fields = decomposer.decompose_global_field(global_field)
        
        # Verify initial state: ghosts should be zero
        p0_field = partitioned_fields[0]
        p1_field = partitioned_fields[1]
        
        # p0 has no top ghost (first partition)
        # p0 bottom ghost (index -1) should be zero initially
        assert p0_field[-1, 0] == 0.0
        
        # p1 top ghost (index 0) should be zero initially
        assert p1_field[0, 0] == 0.0
        
        # Exchange ghosts
        decomposer.exchange_ghosts(partitioned_fields)
        
        # After exchange:
        # p0's bottom ghost should equal p1's top owned row
        # p1's top ghost should equal p0's bottom owned row
        
        p0_bottom_owned = decomposer.partitions[0].get_bottom_boundary_row(p0_field)
        p1_top_ghost = p1_field[0, :]
        assert torch.allclose(p1_top_ghost, p0_bottom_owned)
        
        p1_top_owned = decomposer.partitions[1].get_top_boundary_row(p1_field)
        p0_bottom_ghost = p0_field[-1, :]
        assert torch.allclose(p0_bottom_ghost, p1_top_owned)


class TestGhostExchange:
    """Tests for ghost cell exchange mechanism."""
    
    def test_ghost_exchange_propagates_changes(self):
        """Test that ghost exchange correctly propagates boundary values."""
        decomposer = StripDecomposer(
            global_grid_size=100,
            num_partitions=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        
        # Create partitioned fields
        global_field = torch.zeros(100, 100)
        partitioned_fields = decomposer.decompose_global_field(global_field)
        
        # Modify a boundary value in partition 0
        modified_value = 42.0
        partitioned_fields[0][
            decomposer.partitions[0].owned_end - 1, :
        ] = modified_value
        
        # Exchange ghosts
        decomposer.exchange_ghosts(partitioned_fields)
        
        # The ghost row in partition 1 should have the modified value
        assert torch.allclose(partitioned_fields[1][0, :], torch.full_like(partitioned_fields[1][0, :], modified_value))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
