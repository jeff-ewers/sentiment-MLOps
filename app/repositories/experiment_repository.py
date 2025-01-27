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
    

        