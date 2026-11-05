"""
Batch Generation Script for SUMND Stochastic Instances

This script generates stochastic SUMND instances for multiple scenario counts,
organizing outputs into separate directories for each scenario count.

Author: Research Project - Benders Integration with ML
Date: March 2026
"""

import subprocess
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_generation(
    input_dir: str,
    base_output_dir: str,
    n_scenarios: int,
    method: str = "Merve_with_Covariate",
    stochastic_type: str = "Demand_Only",
    seed: int = 42,
    sigma_L: float = 0.10,
    sigma_U: float = 0.30,
    n_covariates: int = 5,
    overwrite: bool = True
):
    """
    Run stochastic instance generation for a specific scenario count.
    
    Args:
        input_dir: Directory containing deterministic .dow files
        base_output_dir: Base output directory
        n_scenarios: Number of scenarios to generate
        method: Generation method
        stochastic_type: Type of stochasticity
        seed: Random seed
        sigma_L: Lower sigma bound
        sigma_U: Upper sigma bound
        n_covariates: Number of covariates
        overwrite: Whether to overwrite existing files
    """
    # Create output directory with clear naming
    output_dir = Path(base_output_dir) / f"S{n_scenarios}_sigma{sigma_L:.2f}-{sigma_U:.2f}_cov{n_covariates}"
    output_dir = output_dir.as_posix()
    
    logger.info("=" * 80)
    logger.info(f"GENERATING INSTANCES WITH {n_scenarios} SCENARIOS")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)
    
    # Build command
    generation_script = Path(__file__).parent / "SUMND_data_generation.py"
    cmd = [
        sys.executable,  # Use current Python interpreter
        str(generation_script),
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--method", method,
        "--type", stochastic_type,
        "--scenarios", str(n_scenarios),
        "--seed", str(seed),
        "--sigma_L", str(sigma_L),
        "--sigma_U", str(sigma_U),
        "--n_covariates", str(n_covariates),
    ]
    
    if overwrite:
        cmd.append("--overwrite")
    
    # Run the generation
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"✓ Successfully generated {n_scenarios}-scenario instances")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to generate {n_scenarios}-scenario instances")
        logger.error(f"Error: {e.stderr}")
        return False


def main():
    """
    Main batch generation function.
    
    Generates stochastic instances for multiple scenario counts:
    - 100 scenarios
    - 300 scenarios
    - 500 scenarios
    - 700 scenarios
    - 900 scenarios
    
    Each scenario count gets its own organized directory.
    """
    # Configuration - paths relative to script location
    script_dir = Path(__file__).parent.parent.parent  # Navigate to project root
    input_dir = str(script_dir / "DATA" / "NE-SP" / "R")
    base_output_dir = str(script_dir / "DATA" / "SUMND" / "R_stochastic")
    
    # Generation parameters
    method = "Merve_with_Covariate"
    stochastic_type = "Demand_Only"
    seed = 42
    sigma_L = 0.10
    sigma_U = 0.30
    n_covariates = 5
    overwrite = True
    
    # Scenario counts to generate
    scenario_counts = [100, 300, 500, 700, 900]
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("BATCH GENERATION: SUMND STOCHASTIC INSTANCES")
    logger.info("=" * 80)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Base output directory: {base_output_dir}")
    logger.info(f"Method: {method}")
    logger.info(f"Type: {stochastic_type}")
    logger.info(f"Scenario counts: {scenario_counts}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Sigma range: [{sigma_L}, {sigma_U}]")
    logger.info(f"Covariates: {n_covariates}")
    logger.info("=" * 80)
    logger.info("")
    
    # Track results
    results = {}
    
    # Generate for each scenario count
    for idx, n_scenarios in enumerate(scenario_counts, 1):
        logger.info(f"\n[{idx}/{len(scenario_counts)}] Starting generation for {n_scenarios} scenarios...")
        
        success = run_generation(
            input_dir=input_dir,
            base_output_dir=base_output_dir,
            n_scenarios=n_scenarios,
            method=method,
            stochastic_type=stochastic_type,
            seed=seed,
            sigma_L=sigma_L,
            sigma_U=sigma_U,
            n_covariates=n_covariates,
            overwrite=overwrite
        )
        
        results[n_scenarios] = success
    
    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("BATCH GENERATION COMPLETE")
    logger.info("=" * 80)
    
    successful = sum(1 for v in results.values() if v)
    failed = len(results) - successful
    
    logger.info(f"Total scenario counts processed: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info("")
    
    for n_scenarios, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"{status} {n_scenarios} scenarios: {'SUCCESS' if success else 'FAILED'}")
    
    logger.info("=" * 80)
    
    # Print directory structure
    logger.info("")
    logger.info("Generated directory structure:")
    base_path = Path(base_output_dir)
    if base_path.exists():
        for item in sorted(base_path.iterdir()):
            if item.is_dir():
                num_files = len(list(item.glob("*.json")))
                logger.info(f"  {item.name}/ ({num_files} files)")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
