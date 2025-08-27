"""
Tests for experiment versioning system.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from ml_research_pipeline.utils.experiment_versioning import (
    ExperimentVersionManager,
    ExperimentComparison
)
from ml_research_pipeline.utils.experiment_manager import ExperimentManager
from ml_research_pipeline.config import ExperimentConfig


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def version_manager(temp_dir):
    """Create experiment version manager."""
    return ExperimentVersionManager(temp_dir)


@pytest.fixture
def test_config():
    """Create test experiment configuration."""
    return ExperimentConfig(
        name="test_experiment",
        description="Test experiment for versioning",
        version="1.0.0",
        seed=42,
        output_dir="test_output",
        deterministic=True
    )


class TestExperimentVersionManager:
    """Test experiment version manager."""
    
    def test_initialization(self, temp_dir):
        """Test version manager initialization."""
        manager = ExperimentVersionManager(temp_dir)
        
        assert manager.base_dir == temp_dir
        assert manager.version_db_file == temp_dir / "experiment_versions.json"
        assert manager.comparison_cache_file == temp_dir / "comparison_cache.json"
        
        # Check initial database structure
        assert "experiments" in manager.version_db
        assert "version_chains" in manager.version_db
        assert "tags" in manager.version_db
    
    def test_experiment_registration(self, version_manager, test_config, temp_dir):
        """Test experiment registration."""
        # Create experiment manager
        exp_manager = ExperimentManager(
            config=test_config,
            base_output_dir=temp_dir / "experiments",
            enable_git_tracking=False
        )
        
        # Register experiment
        tags = ["baseline", "test"]
        version_manager.register_experiment(exp_manager, tags=tags)
        
        # Check registration
        exp_id = exp_manager.experiment_id
        assert exp_id in version_manager.version_db["experiments"]
        
        experiment_entry = version_manager.version_db["experiments"][exp_id]
        assert experiment_entry["name"] == test_config.name
        assert experiment_entry["version"] == test_config.version
        assert experiment_entry["tags"] == tags
        assert experiment_entry["parent_id"] is None
        assert experiment_entry["children_ids"] == []
        
        # Check version chain
        base_name = test_config.name.split('_v')[0]
        assert base_name in version_manager.version_db["version_chains"]
        assert len(version_manager.version_db["version_chains"][base_name]) == 1
        
        # Check tags
        for tag in tags:
            assert tag in version_manager.version_db["tags"]
            assert exp_id in version_manager.version_db["tags"][tag]
    
    def test_parent_child_relationships(self, version_manager, test_config, temp_dir):
        """Test parent-child experiment relationships."""
        # Create parent experiment
        parent_config = test_config.update(name="parent_experiment", version="1.0")
        parent_manager = ExperimentManager(
            config=parent_config,
            base_output_dir=temp_dir / "experiments",
            enable_git_tracking=False
        )
        
        version_manager.register_experiment(parent_manager)
        parent_id = parent_manager.experiment_id
        
        # Create child experiment
        child_config = test_config.update(name="child_experiment", version="1.1")
        child_manager = ExperimentManager(
            config=child_config,
            base_output_dir=temp_dir / "experiments",
            enable_git_tracking=False
        )
        
        version_manager.register_experiment(child_manager, parent_experiment_id=parent_id)
        child_id = child_manager.experiment_id
        
        # Check relationships
        parent_entry = version_manager.version_db["experiments"][parent_id]
        child_entry = version_manager.version_db["experiments"][child_id]
        
        assert child_entry["parent_id"] == parent_id
        assert child_id in parent_entry["children_ids"]
    
    def test_experiment_lineage(self, version_manager, test_config, temp_dir):
        """Test experiment lineage retrieval."""
        # Create experiment chain: grandparent -> parent -> child
        experiments = []
        
        for i, name in enumerate(["grandparent", "parent", "child"]):
            config = test_config.update(name=f"{name}_experiment", version=f"1.{i}")
            manager = ExperimentManager(
                config=config,
                base_output_dir=temp_dir / "experiments",
                enable_git_tracking=False
            )
            
            parent_id = experiments[-1].experiment_id if experiments else None
            version_manager.register_experiment(manager, parent_experiment_id=parent_id)
            experiments.append(manager)
        
        # Test lineage for child (should have 2 ancestors)
        child_id = experiments[2].experiment_id
        lineage = version_manager.get_experiment_lineage(child_id)
        
        assert lineage["lineage_depth"] == 2
        assert len(lineage["ancestors"]) == 2
        assert len(lineage["descendants"]) == 0
        
        # Test lineage for grandparent (should have 2 descendants)
        grandparent_id = experiments[0].experiment_id
        lineage = version_manager.get_experiment_lineage(grandparent_id)
        
        assert lineage["lineage_depth"] == 0
        assert len(lineage["ancestors"]) == 0
        assert lineage["total_descendants"] == 2
    
    def test_experiment_comparison(self, version_manager, test_config, temp_dir):
        """Test experiment comparison."""
        # Create two experiments with different configurations
        config1 = test_config.update(name="exp1", seed=42)
        config2 = test_config.update(name="exp2", seed=123)
        
        manager1 = ExperimentManager(
            config=config1,
            base_output_dir=temp_dir / "experiments",
            enable_git_tracking=False
        )
        
        manager2 = ExperimentManager(
            config=config2,
            base_output_dir=temp_dir / "experiments",
            enable_git_tracking=False
        )
        
        # Complete experiments with different metrics
        manager1.start_experiment()
        manager1.complete_experiment({"accuracy": 0.9, "loss": 0.1})
        
        manager2.start_experiment()
        manager2.complete_experiment({"accuracy": 0.85, "loss": 0.15})
        
        # Register experiments
        version_manager.register_experiment(manager1)
        version_manager.register_experiment(manager2)
        
        # Compare experiments
        comparison = version_manager.compare_experiments(
            manager1.experiment_id, 
            manager2.experiment_id
        )
        
        assert comparison.experiment_1_id == manager1.experiment_id
        assert comparison.experiment_2_id == manager2.experiment_id
        assert comparison.reproducibility_status == "different"  # Different seeds
        
        # Check config differences
        assert "seed" in comparison.config_differences
        assert comparison.config_differences["seed"] == (42, 123)
        
        # Check metric differences
        assert "accuracy" in comparison.metric_differences
        assert "loss" in comparison.metric_differences
        
        accuracy_diff = comparison.metric_differences["accuracy"]
        assert accuracy_diff["value_1"] == 0.9
        assert accuracy_diff["value_2"] == 0.85
        assert abs(accuracy_diff["absolute_difference"] - 0.05) < 1e-10
    
    def test_similar_experiments_search(self, version_manager, test_config, temp_dir):
        """Test finding similar experiments."""
        # Create multiple experiments with varying similarity
        experiments = []
        
        # Very similar experiments (same config, different seeds)
        for i in range(3):
            config = test_config.update(name=f"similar_exp_{i}", seed=42 + i)
            manager = ExperimentManager(
                config=config,
                base_output_dir=temp_dir / "experiments",
                enable_git_tracking=False
            )
            manager.start_experiment()
            manager.complete_experiment({"accuracy": 0.9 + i * 0.01})
            version_manager.register_experiment(manager)
            experiments.append(manager)
        
        # Different experiment
        different_config = test_config.update(
            name="different_exp", 
            seed=999,
            description="Very different experiment"
        )
        different_manager = ExperimentManager(
            config=different_config,
            base_output_dir=temp_dir / "experiments",
            enable_git_tracking=False
        )
        different_manager.start_experiment()
        different_manager.complete_experiment({"accuracy": 0.5})
        version_manager.register_experiment(different_manager)
        
        # Find similar experiments
        reference_id = experiments[0].experiment_id
        similar_experiments = version_manager.find_similar_experiments(
            reference_id, 
            similarity_threshold=0.8
        )
        
        # Should find the other similar experiments but not the different one
        similar_ids = [exp_id for exp_id, score in similar_experiments]
        
        assert experiments[1].experiment_id in similar_ids
        assert experiments[2].experiment_id in similar_ids
        assert different_manager.experiment_id not in similar_ids
    
    def test_version_chains(self, version_manager, test_config, temp_dir):
        """Test version chain management."""
        base_name = "test_experiment"
        versions = ["1.0", "1.1", "2.0"]
        
        # Create experiments with different versions
        for version in versions:
            config = test_config.update(name=f"{base_name}_v{version}", version=version)
            manager = ExperimentManager(
                config=config,
                base_output_dir=temp_dir / "experiments",
                enable_git_tracking=False
            )
            version_manager.register_experiment(manager)
        
        # Get version chain
        version_chain = version_manager.get_version_chain(base_name)
        
        assert len(version_chain) == 3
        
        # Check ordering (should be sorted by timestamp)
        chain_versions = [entry["version"] for entry in version_chain]
        assert chain_versions == versions  # Should maintain order due to sequential creation
    
    def test_tags_functionality(self, version_manager, test_config, temp_dir):
        """Test experiment tagging functionality."""
        tags_experiments = [
            (["baseline", "cpu"], "baseline_exp"),
            (["baseline", "gpu"], "gpu_baseline_exp"),
            (["improved", "gpu"], "improved_exp"),
            (["final"], "final_exp")
        ]
        
        experiment_ids = []
        
        # Create tagged experiments
        for tags, name in tags_experiments:
            config = test_config.update(name=name)
            manager = ExperimentManager(
                config=config,
                base_output_dir=temp_dir / "experiments",
                enable_git_tracking=False
            )
            version_manager.register_experiment(manager, tags=tags)
            experiment_ids.append(manager.experiment_id)
        
        # Test tag queries
        baseline_experiments = version_manager.get_experiments_by_tag("baseline")
        assert len(baseline_experiments) == 2
        assert experiment_ids[0] in baseline_experiments
        assert experiment_ids[1] in baseline_experiments
        
        gpu_experiments = version_manager.get_experiments_by_tag("gpu")
        assert len(gpu_experiments) == 2
        assert experiment_ids[1] in gpu_experiments
        assert experiment_ids[2] in gpu_experiments
        
        final_experiments = version_manager.get_experiments_by_tag("final")
        assert len(final_experiments) == 1
        assert experiment_ids[3] in final_experiments
    
    def test_experiment_report_generation(self, version_manager, test_config, temp_dir):
        """Test comprehensive experiment report generation."""
        # Create experiment with lineage and tags
        config = test_config.update(name="report_test_exp")
        manager = ExperimentManager(
            config=config,
            base_output_dir=temp_dir / "experiments",
            enable_git_tracking=False
        )
        
        manager.start_experiment()
        manager.complete_experiment({"accuracy": 0.95, "loss": 0.05})
        
        version_manager.register_experiment(manager, tags=["test", "report"])
        
        # Generate report
        report = version_manager.generate_experiment_report(manager.experiment_id)
        
        assert "experiment_info" in report
        assert "lineage" in report
        assert "similar_experiments" in report
        assert "version_chain" in report
        assert "tags" in report
        assert "reproducibility_analysis" in report
        assert "generated_at" in report
        
        # Check experiment info
        exp_info = report["experiment_info"]
        assert exp_info["experiment_id"] == manager.experiment_id
        assert exp_info["name"] == config.name
        assert exp_info["tags"] == ["test", "report"]
    
    def test_comparison_caching(self, version_manager, test_config, temp_dir):
        """Test comparison result caching."""
        # Create two experiments
        config1 = test_config.update(name="cache_test_1")
        config2 = test_config.update(name="cache_test_2")
        
        manager1 = ExperimentManager(
            config=config1,
            base_output_dir=temp_dir / "experiments",
            enable_git_tracking=False
        )
        
        manager2 = ExperimentManager(
            config=config2,
            base_output_dir=temp_dir / "experiments",
            enable_git_tracking=False
        )
        
        manager1.start_experiment()
        manager1.complete_experiment({"accuracy": 0.9})
        
        manager2.start_experiment()
        manager2.complete_experiment({"accuracy": 0.85})
        
        version_manager.register_experiment(manager1)
        version_manager.register_experiment(manager2)
        
        # First comparison (should be computed)
        comparison1 = version_manager.compare_experiments(
            manager1.experiment_id, 
            manager2.experiment_id,
            use_cache=False
        )
        
        # Check that result is cached
        cache_key = f"{manager1.experiment_id}_{manager2.experiment_id}"
        assert cache_key in version_manager.comparison_cache
        
        # Second comparison (should use cache)
        comparison2 = version_manager.compare_experiments(
            manager1.experiment_id, 
            manager2.experiment_id,
            use_cache=True
        )
        
        # Results should be identical
        assert comparison1.similarity_score == comparison2.similarity_score
        assert comparison1.reproducibility_status == comparison2.reproducibility_status
    
    def test_cleanup_old_comparisons(self, version_manager):
        """Test cleanup of old comparison cache entries."""
        # Add some mock comparison entries with old timestamps
        from datetime import datetime, timedelta
        
        old_timestamp = (datetime.now() - timedelta(days=35)).isoformat()
        recent_timestamp = (datetime.now() - timedelta(days=5)).isoformat()
        
        version_manager.comparison_cache = {
            "old_comparison_1": {"timestamp": old_timestamp, "data": "old1"},
            "old_comparison_2": {"timestamp": old_timestamp, "data": "old2"},
            "recent_comparison": {"timestamp": recent_timestamp, "data": "recent"}
        }
        
        # Cleanup entries older than 30 days
        version_manager.cleanup_old_comparisons(days_old=30)
        
        # Check that old entries are removed
        assert "old_comparison_1" not in version_manager.comparison_cache
        assert "old_comparison_2" not in version_manager.comparison_cache
        assert "recent_comparison" in version_manager.comparison_cache
    
    def test_database_export_import(self, version_manager, test_config, temp_dir):
        """Test database export functionality."""
        # Create some test data
        config = test_config.update(name="export_test")
        manager = ExperimentManager(
            config=config,
            base_output_dir=temp_dir / "experiments",
            enable_git_tracking=False
        )
        
        version_manager.register_experiment(manager, tags=["export"])
        
        # Export database
        export_file = temp_dir / "exported_db.json"
        version_manager.export_version_database(export_file)
        
        assert export_file.exists()
        
        # Check export content
        with open(export_file, 'r') as f:
            export_data = json.load(f)
        
        assert "version_db" in export_data
        assert "comparison_cache" in export_data
        assert "exported_at" in export_data
        assert "total_experiments" in export_data
        assert "total_comparisons" in export_data
        
        # Verify experiment data is included
        assert manager.experiment_id in export_data["version_db"]["experiments"]


class TestExperimentComparison:
    """Test experiment comparison data structure."""
    
    def test_comparison_creation(self):
        """Test comparison object creation and serialization."""
        comparison = ExperimentComparison(
            experiment_1_id="exp1",
            experiment_2_id="exp2",
            config_differences={"seed": (42, 123)},
            metric_differences={"accuracy": {"value_1": 0.9, "value_2": 0.85, "absolute_difference": 0.05}},
            reproducibility_status="different",
            similarity_score=0.8,
            timestamp="2024-01-01T00:00:00"
        )
        
        # Test serialization
        comparison_dict = comparison.to_dict()
        
        assert comparison_dict["experiment_1_id"] == "exp1"
        assert comparison_dict["experiment_2_id"] == "exp2"
        assert "config_differences" in comparison_dict
        assert "metric_differences" in comparison_dict
        assert "reproducibility_status" in comparison_dict
        assert "similarity_score" in comparison_dict
    
    def test_comparison_with_significance_matrix(self):
        """Test comparison object creation and serialization."""
        comparison = ExperimentComparison(
            experiment_1_id="exp1",
            experiment_2_id="exp2",
            config_differences={},
            metric_differences={},
            reproducibility_status="similar",
            similarity_score=0.9,
            timestamp="2024-01-01T00:00:00"
        )
        
        # Test serialization
        comparison_dict = comparison.to_dict()
        
        assert comparison_dict["experiment_1_id"] == "exp1"
        assert comparison_dict["experiment_2_id"] == "exp2"
        assert comparison_dict["reproducibility_status"] == "similar"
        assert comparison_dict["similarity_score"] == 0.9