import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from ..constants import TileSelection, Timer
from ..io.h5 import get_features
from ..thirdparty.hloc.extractors.superpoint import SuperPoint
from ..thirdparty.LightGlue.lightglue import LightGlue
from .matcher_base import DetectorFreeMatcherBase
from .lightglue import featuresDict2Lightglue, rbd

logger = logging.getLogger("dim")


def rotate_image_90(image: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Rotate image by 90 degrees k times.
    
    Args:
        image (np.ndarray): Input image
        k (int): Number of 90-degree rotations (1=90, 2=180, 3=270)
    
    Returns:
        np.ndarray: Rotated image
    """
    return np.rot90(image, k)


def rotate_keypoints_90(keypoints: np.ndarray, image_shape: tuple, k: int = 1) -> np.ndarray:
    """
    Rotate keypoints by 90 degrees k times to match np.rot90(image, k) (CCW).

    np.rot90(image, k=1) rotates counter-clockwise: pixel at (x, y) in image
    of shape (H, W) maps to (y, W-1-x) in the rotated image of shape (W, H).

    Args:
        keypoints (np.ndarray): Keypoints (N, 2) in [x, y] format.
        image_shape (tuple): Shape (H, W) of the image the keypoints live in.
        k (int): Number of CCW 90-degree rotations (1=90°, 2=180°, 3=270°).

    Returns:
        np.ndarray: Rotated keypoints (N, 2).
    """
    H, W = image_shape
    rotated_kpts = keypoints.copy().astype(np.float32)

    k = k % 4

    for _ in range(k):
        # CCW 90°: (x, y) → (y, W-1-x), new image shape becomes (W, H)
        # Use explicit copies of x and y to avoid in-place aliasing bugs
        x = rotated_kpts[:, 0].copy()
        y = rotated_kpts[:, 1].copy()
        rotated_kpts[:, 0] = y
        rotated_kpts[:, 1] = W - 1 - x
        H, W = W, H

    return rotated_kpts


def rotate_keypoints_back(keypoints: np.ndarray, original_shape: tuple, k: int = 1) -> np.ndarray:
    """
    Rotate keypoints back to original orientation from k 90-degree rotations.
    
    Args:
        keypoints (np.ndarray): Keypoints in rotated image coordinates
        original_shape (tuple): Original image shape (H, W)
        k (int): Number of 90-degree rotations that were applied
    
    Returns:
        np.ndarray: Keypoints back in original coordinates
    """
    # To undo k rotations, rotate by (4 - k) % 4
    H, W = original_shape
    
    # Calculate the shape after k rotations
    rotated_shape = (W, H) if k % 2 == 1 else (H, W)
    
    return rotate_keypoints_90(keypoints, rotated_shape, (4 - k) % 4)


class SuperPointLightGluePairMatcher(DetectorFreeMatcherBase):
    """
    SuperPoint + LightGlue Pair Matcher with rotation invariance.
    
    This matcher addresses the non-rotation-invariant nature of SuperPoint+LightGlue
    by running matching on the original image and 3 rotated versions (90, 180, 270 degrees)
    of the second image, then selecting matches from the pair with the most matches.
    """

    _default_conf = {
        "sp_cfg": {
            "nms_radius": 5,
            "max_keypoints": 4000,
            "keypoint_threshold": 0.005,
        },
        "lg_cfg": {
            "features": "superpoint",
            "n_layers": 9,
            "depth_confidence": 0.9,
            "width_confidence": 0.95,
            "filter_threshold": 0.3,
            "flash": True,
        },
    }
    
    grayscale = False
    as_float = True
    min_matches = 20
    min_matches_per_tile = 3
    max_tile_size = 1200

    def __init__(self, config={}) -> None:
        """
        Initialize SuperPointLightGluePairMatcher.
        
        Args:
            config: Configuration options (Config object).
        """
        super().__init__(config)

        # Get configuration
        sp_cfg = self.config["matcher"].get("sp_cfg", self._default_conf["sp_cfg"])
        lg_cfg = self.config["matcher"].get("lg_cfg", self._default_conf["lg_cfg"])

        # Initialize SuperPoint extractor
        self.extractor = SuperPoint(sp_cfg).eval().to(self._device)

        # Initialize LightGlue matcher
        self.matcher = LightGlue(**lg_cfg).eval().to(self._device)

    def _extract_features(self, image: np.ndarray) -> dict:
        """
        Extract features from image using SuperPoint.
        
        Args:
            image (np.ndarray): Input image
        
        Returns:
            dict: Features dictionary with keypoints and descriptors (numpy)
        """
        # Convert image to tensor
        img_tensor = self._frame2tensor(image, self._device)
        
        # Extract features via SuperPoint (expects dict with 'image' key)
        with torch.inference_mode():
            features = self.extractor({"image": img_tensor})
        
        # Remove batch dimension and list/tuple wrapping
        feats = {k: v[0] if isinstance(v, (list, tuple)) else v for k, v in features.items()}
        
        # Convert tensors to numpy (ensure they're on CPU first)
        feats_np = {}
        for k, v in feats.items():
            if isinstance(v, torch.Tensor):
                # Move to CPU and convert to numpy
                feats_np[k] = v.detach().cpu().numpy()
            else:
                feats_np[k] = np.asarray(v) if not isinstance(v, np.ndarray) else v
        
        # Fix descriptor layout: ensure (N, D) format where N = num keypoints
        if "descriptors" in feats_np:
            desc = feats_np["descriptors"]
            kpts = feats_np["keypoints"]
            n_kpts = len(kpts)
            
            if desc.ndim == 2:
                if desc.shape[1] == n_kpts and desc.shape[0] != n_kpts:
                    # (D, N) -> transpose to (N, D)
                    feats_np["descriptors"] = desc.T
        
        return feats_np

    def _match_with_features(self, feats0: dict, feats1: dict) -> tuple:
        """
        Match features between two sets of extracted features.
        Uses the same pattern as LightGlueMatcher._match_pairs().

        Args:
            feats0 (dict): Features from image 0 (numpy arrays)
            feats1 (dict): Features from image 1 (numpy arrays)

        Returns:
            tuple: (matched_keypoints_0, matched_keypoints_1, num_matches)
        """
        try:
            # Convert to LightGlue format: fixes descriptor layout, adds batch dim, moves to device
            feats0_lg = featuresDict2Lightglue(feats0, self._device)
            feats1_lg = featuresDict2Lightglue(feats1, self._device)

            # Run matcher
            with torch.inference_mode():
                match_res = self.matcher({"image0": feats0_lg, "image1": feats1_lg})

            # Remove batch dim from all outputs (same as lightglue.py)
            _, _, matches01 = [rbd(x) for x in [feats0_lg, feats1_lg, match_res]]

            # Convert all tensors to numpy (same as lightglue.py)
            match_res_np = {
                k: v.cpu().numpy()
                for k, v in matches01.items()
                if isinstance(v, torch.Tensor)
            }

            # matches shape: (M, 2) — pairs of (idx_in_img0, idx_in_img1)
            matches_idx = match_res_np["matches"]

            if len(matches_idx) == 0:
                return np.array([], dtype=np.float32).reshape(0, 2), \
                       np.array([], dtype=np.float32).reshape(0, 2), 0

            mkpts0 = feats0["keypoints"][matches_idx[:, 0]]
            mkpts1 = feats1["keypoints"][matches_idx[:, 1]]

            return mkpts0, mkpts1, len(matches_idx)

        except Exception as e:
            logger.warning(f"Matching failed: {str(e)}")
            logger.debug("", exc_info=True)
            return np.array([], dtype=np.float32).reshape(0, 2), \
                   np.array([], dtype=np.float32).reshape(0, 2), 0

    @torch.no_grad()
    def _match_pairs(
        self,
        feature_path: Path,
        img0_path: Path,
        img1_path: Path,
    ):
        """
        Perform pair matching with rotation compensation.
        
        Args:
            feature_path (Path): Path to h5 feature file
            img0_path (Path): Path to first image
            img1_path (Path): Path to second image
        
        Returns:
            np.ndarray: Matching array
        """
        img0_name = img0_path.name
        img1_name = img1_path.name

        # Load images
        image0 = self._load_image_np(img0_path)
        image1 = self._load_image_np(img1_path)

        # Resize images if needed
        image0_ = self._resize_image(self._quality, image0)
        image1_ = self._resize_image(self._quality, image1)

        original_shape_0 = image0_.shape[:2]
        original_shape_1 = image1_.shape[:2]

        # Extract features from image 0
        logger.debug(f"Extracting features from {img0_name}")
        feats0 = self._extract_features(image0_)

        # Try matching with original and rotated versions of image 1
        best_mkpts0 = None
        best_mkpts1 = None
        best_rotation = 0
        best_count = 0

        for rotation in [0, 1, 2, 3]:  # 0, 90, 180, 270 degrees
            # Rotate image 1
            if rotation == 0:
                image1_rot = image1_
                shape_1_rot = original_shape_1
            else:
                image1_rot = rotate_image_90(image1_, rotation)
                shape_1_rot = image1_rot.shape[:2]

            logger.debug(f"  Extracting features from {img1_name} (rotation {rotation * 90}°)")
            # Extract features from rotated image 1
            feats1_rot = self._extract_features(image1_rot)

            # Match features
            mkpts0, mkpts1_rot, match_count = self._match_with_features(feats0, feats1_rot)

            # If this rotation gives more matches, update best
            if match_count > best_count:
                best_count = match_count
                best_mkpts0 = mkpts0
                best_mkpts1 = mkpts1_rot
                best_rotation = rotation

            logger.debug(f"    Rotation {rotation * 90}° degrees: {match_count} matches")

        if best_count == 0:
            logger.warning(f"No matches found between {img0_name} and {img1_name}")
            matches = np.array([], dtype=np.int32).reshape(0, 2)
            return matches

        # Rotate matched keypoints back to original orientation if needed
        if best_rotation != 0:
            best_mkpts1 = rotate_keypoints_back(best_mkpts1, original_shape_1, best_rotation)

        # Resize keypoints back to original image scale
        best_mkpts0 = self._resize_keypoints(self._quality, best_mkpts0)
        best_mkpts1 = self._resize_keypoints(self._quality, best_mkpts1)

        # Create 1-to-1 matching array
        matches0 = np.arange(best_mkpts0.shape[0])
        matches = np.hstack((matches0.reshape((-1, 1)), matches0.reshape((-1, 1))))

        # Update features in h5 file
        matches = self._update_features_h5(
            feature_path,
            img0_name,
            img1_name,
            best_mkpts0,
            best_mkpts1,
            matches,
        )

        logger.debug(f"Best rotation: {best_rotation * 90}° with {best_count} matches")

        return matches

    def _match_by_tile(
        self,
        feature_path: Path,
        img0: Path,
        img1: Path,
        method: TileSelection = TileSelection.PRESELECTION,
        select_unique: bool = True,
    ) -> np.ndarray:
        """
        Match features between two images using a tiling approach.
        
        Note: For this matcher, tiling is simplified to not split the tiles further.
        """
        logger.debug("Matching by tile...")

        # For now, just do full image matching
        # In future, this could be extended to handle tiling better
        matches = self._match_pairs(feature_path, img0, img1)

        return matches if matches is not None else np.array([], dtype=np.int32).reshape(0, 2)

    def _frame2tensor(self, image: np.ndarray, device: str = "cpu") -> torch.Tensor:
        """
        Convert image to tensor suitable for SuperPoint.
        
        Args:
            image (np.ndarray): Input image
            device (str): Device to move tensor to
        
        Returns:
            torch.Tensor: Image tensor with shape (1, 1, H, W)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions: (H, W) -> (1, 1, H, W)
        image = torch.from_numpy(image[None, None]).to(device)
        
        return image
