from moe.model import MoEModel
from moe.trainer import Trainer
from diffusers.pipeline import DiffusersPipeline

def main():
    # Initialize the MoE model
    moe_model = MoEModel()
    
    # Initialize the Trainer
    trainer = Trainer(moe_model)
    
    # Train the model
    trainer.train()
    
    # Initialize the Diffusers pipeline
    pipeline = DiffusersPipeline()
    
    # Run inference
    results = pipeline.run_inference()
    
    # Print results
    print(results)

if __name__ == "__main__":
    main()