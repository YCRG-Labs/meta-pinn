"""
Advanced experiment versioning and comparison system.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

from .logging_utils import LoggerMixin
from .experiment_manager import ExperimentManager, ExperimentMetadata


@dataclass
class ExperimentComparison:
    """Results of comparing two experiments."""
    
    experiment_1_id: str
    experiment_2_id: str
    config_differences: Dict[str, Tuple[Any, Any]]
    metric_differences: Dict[str, Dict[str, float]]
    reproducibility_status: str  # identical, similar, different
    similarity_score: float
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ExperimentVersionManager(LoggerMixin):
    """Advanced experiment versioning and comparison system."""
    
    def __init__(self, base_dir: Path):
        """Initialize experiment version manager.
        
        Args:
            base_dir: Base directory containing experiments
        """
        self.base_dir = Path(base_dir)
        self.version_db_file = self.base_dir / "experiment_versions.json"
        self.comparison_cache_file = self.base_dir / "comparison_cache.json"
        
        # Load existing data
        self.version_db = self._load_version_db()
        self.comparison_cache = self._load_comparison_cache()
    
    def _load_version_db(self) -> Dict[str, Any]:
        """Load experiment version database."""
        if self.version_db_file.exists():
            with open(self.version_db_file, 'r') as f:
                return json.load(f)
        return {
            "experiments": {},
            "version_chains": {},
            "tags": {}
        }
    
    def _save_version_db(self):
        """Save experiment version database."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with open(self.version_db_file, 'w') as f:
            json.dump(self.version_db, f, indent=2, default=str)
    
    def _load_comparison_cache(self) -> Dict[str, Any]:
        """Load comparison cache."""
        if self.comparison_cache_file.exists():
            with open(self.comparison_cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_comparison_cache(self):
        """Save comparison cache."""
        with open(self.comparison_cache_file, 'w') as f:
            json.dump(self.comparison_cache, f, indent=2, default=str)
    
    def register_experiment(
        self, 
        manager: ExperimentManager,
        parent_experiment_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        """Register experiment in version system.
        
        Args:
            manager: Experiment manager
            parent_experiment_id: ID of parent experiment (for versioning)
            tags: Tags for categorizing experiment
        """
        experiment_id = manager.experiment_id
        
        # Create experiment entry
        experiment_entry = {
            "experiment_id": experiment_id,
            "name": manager.config.name,
            "version": manager.config.version,
            "timestamp": manager.metadata.timestamp,
            "config_hash": manager.metadata.config_hash,
            "experiment_dir": str(manager.experiment_dir),
            "parent_id": parent_experiment_id,
            "children_ids": [],
            "tags": tags or [],
            "status": manager.metadata.status,
            "final_metrics": manager.metadata.final_metrics
        }
        
        # Add to database
        self.version_db["experiments"][experiment_id] = experiment_entry
        
        # Update parent-child relationships
        if parent_experiment_id and parent_experiment_id in self.version_db["experiments"]:
            self.version_db["experiments"][parent_experiment_id]["children_ids"].append(experiment_id)
        
        # Update version chains
        base_name = manager.config.name.split('_v')[0]  # Remove version suffix
        if base_name not in self.version_db["version_chains"]:
            self.version_db["version_chains"][base_name] = []
        
        self.version_db["version_chains"][base_name].append({
            "experiment_id": experiment_id,
            "version": manager.config.version,
            "timestamp": manager.metadata.timestamp
        })
        
        # Sort version chain by timestamp
        self.version_db["version_chains"][base_name].sort(key=lambda x: x["timestamp"])
        
        # Update tags
        for tag in (tags or []):
            if tag not in self.version_db["tags"]:
                self.version_db["tags"][tag] = []
            self.version_db["tags"][tag].append(experiment_id)
        
        self._save_version_db()
        self.log_info(f"Registered experiment {experiment_id} in version system")
    
    def get_experiment_lineage(self, experiment_id: str) -> Dict[str, Any]:
        """Get complete lineage of an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment lineage information
        """
        if experiment_id not in self.version_db["experiments"]:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.version_db["experiments"][experiment_id]
        
        # Get ancestors
        ancestors = []
        current_id = experiment["parent_id"]
        while current_id and current_id in self.version_db["experiments"]:
            ancestors.append(self.version_db["experiments"][current_id])
            current_id = self.version_db["experiments"][current_id]["parent_id"]
        
        # Get descendants (recursive)
        def get_descendants(exp_id):
            descendants = []
            exp = self.version_db["experiments"][exp_id]
            for child_id in exp["children_ids"]:
                if child_id in self.version_db["experiments"]:
                    child_exp = self.version_db["experiments"][child_id]
                    descendants.append(child_exp)
                    descendants.extend(get_descendants(child_id))
            return descendants
        
        descendants = get_descendants(experiment_id)
        
        return {
            "experiment": experiment,
            "ancestors": ancestors,
            "descendants": descendants,
            "lineage_depth": len(ancestors),
            "total_descendants": len(descendants)
        }
    
    def compare_experiments(
        self, 
        experiment_id_1: str, 
        experiment_id_2: str,
        use_cache: bool = True
    ) -> ExperimentComparison:
        """Compare two experiments.
        
        Args:
            experiment_id_1: First experiment ID
            experiment_id_2: Second experiment ID
            use_cache: Whether to use cached comparison results
            
        Returns:
            Experiment comparison results
        """
        # Check cache first
        cache_key = f"{experiment_id_1}_{experiment_id_2}"
        reverse_cache_key = f"{experiment_id_2}_{experiment_id_1}"
        
        if use_cache and (cache_key in self.comparison_cache or reverse_cache_key in self.comparison_cache):
            cached_result = self.comparison_cache.get(cache_key) or self.comparison_cache.get(reverse_cache_key)
            return ExperimentComparison(**cached_result)
        
        # Load experiments
        exp1 = self.version_db["experiments"][experiment_id_1]
        exp2 = self.version_db["experiments"][experiment_id_2]
        
        # Load experiment managers for detailed comparison
        manager1 = ExperimentManager.load_experiment(exp1["experiment_dir"])
        manager2 = ExperimentManager.load_experiment(exp2["experiment_dir"])
        
        # Compare configurations
        config1 = manager1.config.to_dict()
        config2 = manager2.config.to_dict()
        
        config_differences = {}
        for key in set(config1.keys()) | set(config2.keys()):
            val1 = config1.get(key)
            val2 = config2.get(key)
            if val1 != val2:
                config_differences[key] = (val1, val2)
        
        # Compare metrics
        metric_differences = {}
        if exp1["final_metrics"] and exp2["final_metrics"]:
            metrics1 = exp1["final_metrics"]
            metrics2 = exp2["final_metrics"]
            
            for metric in set(metrics1.keys()) | set(metrics2.keys()):
                if metric in metrics1 and metric in metrics2:
                    val1, val2 = metrics1[metric], metrics2[metric]
                    diff = abs(val1 - val2)
                    rel_diff = diff / max(abs(val1), abs(val2), 1e-10)
                    
                    metric_differences[metric] = {
                        "value_1": val1,
                        "value_2": val2,
                        "absolute_difference": diff,
                        "relative_difference": rel_diff
                    }
        
        # Determine reproducibility status
        reproducibility_status = self._determine_reproducibility_status(
            manager1, manager2, config_differences, metric_differences
        )
        
        # Calculate similarity score
        similarity_score = self._calculate_similarity_score(
            config_differences, metric_differences
        )
        
        # Create comparison result
        comparison = ExperimentComparison(
            experiment_1_id=experiment_id_1,
            experiment_2_id=experiment_id_2,
            config_differences=config_differences,
            metric_differences=metric_differences,
            reproducibility_status=reproducibility_status,
            similarity_score=similarity_score,
            timestamp=datetime.now().isoformat()
        )
        
        # Cache result
        self.comparison_cache[cache_key] = comparison.to_dict()
        self._save_comparison_cache()
        
        return comparison
    
    def _determine_reproducibility_status(
        self,
        manager1: ExperimentManager,
        manager2: ExperimentManager,
        config_differences: Dict[str, Tuple[Any, Any]],
        metric_differences: Dict[str, Dict[str, float]]
    ) -> str:
        """Determine reproducibility status between experiments."""
        # Check if configurations are identical (ignoring timestamp fields)
        significant_config_diffs = {
            k: v for k, v in config_differences.items()
            if k not in ['timestamp', 'experiment_id', 'output_dir']
        }
        
        if not significant_config_diffs:
            # Same configuration - check metric similarity
            if not metric_differences:
                return "identical"
            
            # Check if metric differences are within tolerance
            max_rel_diff = max(
                (md["relative_difference"] for md in metric_differences.values()),
                default=0
            )
            
            if max_rel_diff < 1e-6:
                return "identical"
            elif max_rel_diff < 0.01:  # 1% tolerance
                return "similar"
            else:
                return "different"
        else:
            # Different configurations
            return "different"
    
    def _calculate_similarity_score(
        self,
        config_differences: Dict[str, Tuple[Any, Any]],
        metric_differences: Dict[str, Dict[str, float]]
    ) -> float:
        """Calculate similarity score between experiments (0-1, higher is more similar)."""
        # Configuration similarity (weight: 0.6)
        config_similarity = 1.0
        if config_differences:
            # Simple heuristic: penalize each difference
            config_similarity = max(0.0, 1.0 - len(config_differences) * 0.1)
        
        # Metric similarity (weight: 0.4)
        metric_similarity = 1.0
        if metric_differences:
            # Average relative differences
            avg_rel_diff = np.mean([
                md["relative_difference"] for md in metric_differences.values()
            ])
            metric_similarity = max(0.0, 1.0 - avg_rel_diff)
        
        # Weighted combination
        overall_similarity = 0.6 * config_similarity + 0.4 * metric_similarity
        return overall_similarity
    
    def find_similar_experiments(
        self,
        experiment_id: str,
        similarity_threshold: float = 0.8,
        max_results: int = 10
    ) -> List[Tuple[str, float]]:
        """Find experiments similar to the given experiment.
        
        Args:
            experiment_id: Reference experiment ID
            similarity_threshold: Minimum similarity score
            max_results: Maximum number of results
            
        Returns:
            List of (experiment_id, similarity_score) tuples
        """
        if experiment_id not in self.version_db["experiments"]:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        similar_experiments = []
        
        for other_id in self.version_db["experiments"]:
            if other_id == experiment_id:
                continue
            
            try:
                comparison = self.compare_experiments(experiment_id, other_id)
                if comparison.similarity_score >= similarity_threshold:
                    similar_experiments.append((other_id, comparison.similarity_score))
            except Exception as e:
                self.log_warning(f"Failed to compare with experiment {other_id}: {e}")
        
        # Sort by similarity score (descending)
        similar_experiments.sort(key=lambda x: x[1], reverse=True)
        
        return similar_experiments[:max_results]
    
    def get_experiments_by_tag(self, tag: str) -> List[str]:
        """Get experiments with specific tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of experiment IDs
        """
        return self.version_db["tags"].get(tag, [])
    
    def get_version_chain(self, base_name: str) -> List[Dict[str, Any]]:
        """Get version chain for experiment base name.
        
        Args:
            base_name: Base experiment name
            
        Returns:
            List of experiments in version chain
        """
        return self.version_db["version_chains"].get(base_name, [])
    
    def generate_experiment_report(self, experiment_id: str) -> Dict[str, Any]:
        """Generate comprehensive report for an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Comprehensive experiment report
        """
        if experiment_id not in self.version_db["experiments"]:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.version_db["experiments"][experiment_id]
        lineage = self.get_experiment_lineage(experiment_id)
        
        # Find similar experiments
        similar_experiments = self.find_similar_experiments(experiment_id, similarity_threshold=0.7)
        
        # Get version chain
        base_name = experiment["name"].split('_v')[0]
        version_chain = self.get_version_chain(base_name)
        
        report = {
            "experiment_info": experiment,
            "lineage": lineage,
            "similar_experiments": similar_experiments,
            "version_chain": version_chain,
            "tags": experiment["tags"],
            "reproducibility_analysis": self._analyze_reproducibility(experiment_id),
            "generated_at": datetime.now().isoformat()
        }
        
        return report
    
    def _analyze_reproducibility(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze reproducibility characteristics of an experiment."""
        experiment = self.version_db["experiments"][experiment_id]
        
        # Load experiment manager
        try:
            manager = ExperimentManager.load_experiment(experiment["experiment_dir"])
            
            # Check if reproducibility validation was performed
            repro_report_file = Path(manager.experiment_dir) / "reproducibility_report.json"
            
            if repro_report_file.exists():
                with open(repro_report_file, 'r') as f:
                    repro_data = json.load(f)
                
                return {
                    "validation_performed": True,
                    "overall_status": repro_data.get("summary", {}).get("overall_status"),
                    "passed_tests": repro_data.get("summary", {}).get("passed_tests", 0),
                    "total_tests": repro_data.get("summary", {}).get("total_tests", 0),
                    "recommendations": repro_data.get("recommendations", [])
                }
            else:
                return {
                    "validation_performed": False,
                    "message": "No reproducibility validation found"
                }
        
        except Exception as e:
            return {
                "validation_performed": False,
                "error": str(e)
            }
    
    def cleanup_old_comparisons(self, days_old: int = 30):
        """Clean up old comparison cache entries.
        
        Args:
            days_old: Remove comparisons older than this many days
        """
        cutoff_date = datetime.now().timestamp() - (days_old * 24 * 3600)
        
        to_remove = []
        for key, comparison in self.comparison_cache.items():
            comparison_date = datetime.fromisoformat(comparison["timestamp"]).timestamp()
            if comparison_date < cutoff_date:
                to_remove.append(key)
        
        for key in to_remove:
            del self.comparison_cache[key]
        
        if to_remove:
            self._save_comparison_cache()
            self.log_info(f"Cleaned up {len(to_remove)} old comparison entries")
    
    def export_version_database(self, output_file: Path):
        """Export version database to file.
        
        Args:
            output_file: Output file path
        """
        export_data = {
            "version_db": self.version_db,
            "comparison_cache": self.comparison_cache,
            "exported_at": datetime.now().isoformat(),
            "total_experiments": len(self.version_db["experiments"]),
            "total_comparisons": len(self.comparison_cache)
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.log_info(f"Exported version database to {output_file}")