import os
import earthaccess
import numpy as np
import matplotlib.pyplot as plt
from georeader.readers import emit
from starcop.models import mag1c_emit
import torch
import omegaconf
import starcop
from starcop.models.model_module import ModelModule
from starcop.models.utils import padding
import georeader
from dotenv import load_dotenv
import json
import time
import logging
import pytorch_lightning as pl
from pytorch_lightning.utilities.migration import pl_legacy_patch
import pytorch_lightning as pl
pl.seed_everything(42)  # For reproducibility
import torch
torch.set_grad_enabled(False) 

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable legacy support
pl_legacy_patch()

# Load environment variables
load_dotenv()

class ModelCache:
    _instance = None
    _model = None
    _config = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelCache()
        return cls._instance

    def get_model(self, model_path, config_path):
        """Load model from local path"""
        if self._model is None:
            logger.info("Loading model from local path...")
            self._model, self._config = load_model_with_emit(model_path, config_path)
            logger.info("Model loaded and cached successfully")
        return self._model, self._config

def download_emit_data(url, output_dir):
    """Download EMIT data from NASA"""
    try:
        os.environ['EARTHDATA_USERNAME'] = os.getenv('EARTHDATA_USERNAME')
        os.environ['EARTHDATA_PASSWORD'] = os.getenv('EARTHDATA_PASSWORD')
        
        if not os.environ.get('EARTHDATA_USERNAME') or not os.environ.get('EARTHDATA_PASSWORD'):
            raise ValueError("Earthdata credentials not found in environment variables.")
        
        earthaccess.login()
        downloaded_files = earthaccess.download(url, output_dir)
        
        if downloaded_files:
            logger.info(f"File downloaded successfully: {downloaded_files[0]}")
            return downloaded_files[0]
        else:
            raise ValueError("No files were downloaded.")
    except Exception as e:
        raise Exception(f"Error downloading data: {str(e)}")

def load_model_with_emit(model_path, config_path):
    """Load the UNET MODEL from local path"""
    try:
        config_general = omegaconf.OmegaConf.load(os.path.join(os.path.dirname(os.path.abspath(starcop.__file__)), 'config.yaml'))
        config_model = omegaconf.OmegaConf.load(config_path)
        config = omegaconf.OmegaConf.merge(config_general, config_model)

        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model = ModelModule(settings=config)
        model.load_state_dict(state_dict, strict=False)
        model.to(torch.device("cpu"))
        model.eval()

        logger.info(f"Model loaded successfully with {model.num_channels} input channels")
        return model, config
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def process_emit_data(emit_id, model_path, config_path, output_dir="static/results"):
    """Main function to process EMIT data and detect methane"""
    product = None
    try:
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs("uploads", exist_ok=True)

        # Check for existing file
        existing_file = None
        for filename in os.listdir("uploads"):
            if emit_id in filename and filename.endswith('.nc'):
                existing_file = os.path.join("uploads", filename)
                break

        if existing_file:
            logger.info(f"Using existing file: {existing_file}")
            product = existing_file
        else:
            logger.info(f"Downloading EMIT data for ID: {emit_id}")
            link = emit.get_radiance_link(emit_id)
            product = download_emit_data(link, "uploads")

        if not product:
            raise Exception("Failed to access EMIT data")

        # Process EMIT data
        rst = emit.EMITImage(product)
        logger.info("EMIT data loaded successfully")

        # Extract file information
        parts = emit_id.split('_')
        timestamp = parts[4] if len(parts) > 4 else 'unknown'
        output_filename = f"methane_detection_{timestamp}.png"
        output_path = os.path.join(output_dir, output_filename)

        # Process RGB bands
        wavelengths_read = np.array([640, 550, 460])
        bands_read = np.argmin(np.abs(wavelengths_read[:, np.newaxis] - rst.wavelengths), axis=1).tolist()
        rst_rgb = rst.read_from_bands(bands_read)
        rgb_raw = rst_rgb.load_raw(transpose=True)

        # Apply MAG1C
        logger.info("Applying MAG1C algorithm...")
        mfoutput, albedo = mag1c_emit.mag1c_emit(rst, column_step=2, georreferenced=False)

        # Load model using cache
        logger.info("Loading local STARCOP model...")
        model_cache = ModelCache.get_instance()
        hsi_model, hsi_config = model_cache.get_model(model_path, config_path)

        # Normalize data
        MAGIC_DIV_BY, RGB_DIV_BY = 240., 20.
        MAGIC_CLIP_TO, RGB_CLIP_TO = [0., 2.], [0., 2.]
        MAGIC_MULT_BY, RGB_MULT_BY = 1750., 60.

        e_mag1c = np.clip(mfoutput / MAGIC_DIV_BY, MAGIC_CLIP_TO[0], MAGIC_CLIP_TO[1]) * MAGIC_MULT_BY
        e_rgb = np.clip(rgb_raw / RGB_DIV_BY, RGB_CLIP_TO[0], RGB_CLIP_TO[1]) * RGB_MULT_BY
        input_data = np.concatenate([e_mag1c[None], e_rgb], axis=0)

        # Run prediction
        logger.info("Running prediction...")
        pred = padding.padded_predict(input_data, model=lambda x: torch.sigmoid(hsi_model(x)))

        # Georeference results
        logger.info("Georeferencing results...")
        crs_utm = georeader.get_utm_epsg(rst.footprint("EPSG:4326"))
        emit_image_utm = rst.to_crs(crs_utm)
        mfgeo = emit_image_utm.georreference(mfoutput, fill_value_default=-1)
        predgeo = emit_image_utm.georreference(pred[0], fill_value_default=0)
        rgbgeo = emit_image_utm.georreference(rgb_raw, fill_value_default=-1)

        # Create visualization
        logger.info("Creating visualization...")
        plt.close('all')
        fig, ax = plt.subplots(2, 2, figsize=(20, 20))

        # RGB plot
        rgbgeomask = np.any(rgbgeo.values == -1, axis=0, keepdims=False)
        rgbplot = (rgbgeo / 12).clip(0, 1)
        rgbplot.values[:, rgbgeomask] = -1
        ax[0, 0].imshow(np.transpose(rgbplot.values, (1, 2, 0)))
        ax[0, 0].set_title("RGB")
        ax[0, 0].axis('off')

        # MAG1C output plot
        mf_data = mfgeo.values
        im1 = ax[0, 1].imshow(mf_data, vmin=0, vmax=1750, cmap='viridis')
        ax[0, 1].set_title("ΔCH4 [ppm x m]")
        ax[0, 1].axis('off')
        plt.colorbar(im1, ax=ax[0, 1], label='Methane Concentration [ppm x m]')

        # Prediction plot
        pred_data = predgeo.values
        im2 = ax[1, 0].imshow(pred_data, vmin=0, vmax=1, cmap='viridis')
        ax[1, 0].set_title("UNET Prediction")
        ax[1, 0].axis('off')
        plt.colorbar(im2, ax=ax[1, 0], label='Probability of Methane Presence')

        # Enhanced Methane Concentration plot
        methane_threshold = 100
        enhanced_mf_data = np.ma.masked_where(mf_data < methane_threshold, mf_data)
        im3 = ax[1, 1].imshow(rgbplot.values.transpose(1, 2, 0), alpha=1.0)
        im4 = ax[1, 1].imshow(enhanced_mf_data, cmap='hot', alpha=0.7, vmin=methane_threshold, vmax=1750)
        ax[1, 1].set_title(f"Methane Concentration Overlay (Threshold: {methane_threshold} ppm x m)")
        ax[1, 1].axis('off')
        plt.colorbar(im4, ax=ax[1, 1], label='Methane Concentration [ppm x m]')

        # Save visualization
        plt.tight_layout()
        logger.info(f"Saving visualization to: {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close('all')

        # Calculate statistics
        methane_threshold = float(100)
        significant_methane = int(np.sum(mf_data > methane_threshold))
        total_pixels = int(mf_data.size)
        percentage_significant = float(significant_methane / total_pixels * 100)
        
        if significant_methane > 0:
            max_concentration = float(np.max(mf_data))
            avg_concentration = float(np.mean(mf_data[mf_data > methane_threshold]))
        else:
            max_concentration = 0.0
            avg_concentration = 0.0

        # Get geographical bounds
        footprint = rst.footprint("EPSG:4326")
        coordinates = footprint.bounds

        # Prepare results
        results = {
            "emit_id": str(emit_id),
            "timestamp": str(timestamp),
            "coordinates": {
                "min_lon": float(coordinates[0]),
                "min_lat": float(coordinates[1]),
                "max_lon": float(coordinates[2]),
                "max_lat": float(coordinates[3])
            },
            "statistics": {
                "significant_methane": int(significant_methane),
                "total_pixels": int(total_pixels),
                "percentage_significant": float(percentage_significant),
                "max_concentration": float(max_concentration),
                "avg_concentration": float(avg_concentration),
                "methane_threshold": float(methane_threshold)
            },
            "has_plumes": bool(significant_methane > 0),
            "visualization_path": str(output_filename),
            "processing_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        logger.info("Processing completed successfully")
        return results

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise

    finally:
        plt.close('all')

if __name__ == "__main__":
    try:
        # Example usage
        test_id = "EMIT_L1B_RAD_001_20240326T214529_2408614_001"
        local_model_path = "path/to/your/model.ckpt"  # Replace with your local model path
        local_config_path = "path/to/your/config.yaml"  # Replace with your local config path
        
        results = process_emit_data(test_id, local_model_path, local_config_path)
        print("\nResults:")
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")
