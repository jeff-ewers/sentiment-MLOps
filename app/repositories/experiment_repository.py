from sqlalchemy.orm import Session
from datetime import datetime
from typing import List, Optional, Dict
from app.database.experiments import Experiment, ExperimentStatus, ExperimentVariant
import random

class ExperimentRepository:
    """Repository for handling A/B testing of model variants"""

    def __init__(self, db: Session):
        self.db = db

    def create_experiment(self,
                          name: str,
                          variants: List[Dict],
                          description: Optional[str] = None,
                          minimum_sample_size: int = 1000,
                          confidence_level: float = 0.95) -> Experiment:
        """Create new experiment with model variants"""

        #validate variant traffic allocation
        total_traffic = sum(variant['traffic_percentage'] for variant in variants)
        if not 0.99 <= total_traffic <= 1.01:
            raise ValueError("Variant traffic percentages must sum to 1.0")
        
        #create experiment
        experiment = Experiment(
            name=name,
            description=description,
            status=ExperimentStatus.DRAFT,
            minimum_sample_size=minimum_sample_size,
            confidence_level=confidence_level
        )
        self.db.add(experiment)
        self.db.flush()

        #create variants
        for variant_data in variants:
            variant = ExperimentVariant(
                experiment_id=experiment.id,
                name=variant_data['name'],
                config=variant_data['config'], #model configuration
                traffic_percentage=variant_data['traffic_percentage'],
                is_control=variant_data.get('is_control', False)
            )
            self.db.add(variant)
        
        self.db.commit()
        self.db.refresh(experiment)
        return experiment
    
    def get_experiment(self, experiment_id: int) -> Optional[Experiment]:
        """Get experiment by ID"""
        return self.db.query(Experiment).filter(Experiment.id == experiment_id).first()
    
    def list_experiments(self, 
                         status: Optional[ExperimentStatus] = None,
                         limit: int = 100) -> List[Experiment]:
        """List experiments with optional status filter"""
        query = self.db.query(Experiment)
        if status:
            query = query.filter(Experiment.status == status)
        return query.order_by(Experiment.created_at.desc()).limit(limit).all()
    
    def update_experiment_status(self,
                                 experiment_id: int,
                                 status: ExperimentStatus) -> Optional[Experiment]:
        """Update experiment status"""
        experiment = self.get_experiment(experiment_id)
        if experiment:
            experiment.status = status
            if status == ExperimentStatus.RUNNING:
                experiment.start_date = datetime.utcnow()
            elif status in [ExperimentStatus.COMPLETED, ExperimentStatus.CANCELLED]:
                experiment.end_date = datetime.utcnow()
            self.db.commit()
            self.db.refresh(experiment)
        return experiment
    
    def get_active_variant(self, experiment_id: int) -> Optional[ExperimentVariant]:
        """Get a random variant based on traffic allocation"""
        experiment = self.get_experiment(experiment_id)
        if not experiment or experiment.status != ExperimentStatus.RUNNING:
            return None
        
        #random assignment based on traffic percentages
        random_value = random.random()
        cumulative_prob = 0

        for variant in experiment.variants:
            cumulative_prob += variant.traffic_percentage
            if random_value <= cumulative_prob:
                return variant
        
        return experiment.variants[0] #fallback to first
    
    def get_experiment_results(self, experiment_id: int) -> Dict:
        """Get statistical results for an experiment"""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return None

        results = {
            'experiment_id': experiment_id,
            'name': experiment.name,
            'status': experiment.status,
            'start_date': experiment.start_date,
            'variants': []
        }

        for variant in experiment.variants:
            variant_metrics = {
                'name': variant.name,
                'is_control': variant.is_control,
                'traffic_percentage': variant.traffic_percentage,
                'predictions': {
                    'total': len(variant.predictions),
                    'positive_ratio': sum(1 for p in variant.predictions if p.sentiment == 'POSITIVE') / len(variant.predictions) if variant.predictions else 0,
                    'confidence': sum(p.confidence for p in variant.predictions) / len(variant.predictions) if variant.predictions else 0
                }
            }
            results['variants'].append(variant_metrics)

        return results