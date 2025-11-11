"""
Firebase Manager - Firebase Integration for FYP-RoleMarket

Provides Firebase Admin SDK integration for:
1. Image storage (Base64 upload to Firebase Storage)
2. Design generation records CRUD (Firestore)
3. Sales prediction records CRUD (Firestore)

Author: Developer (James)
Date: 2025-11-11
Version: 1.0
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import os
import base64
import io
from datetime import datetime
from PIL import Image

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, firestore, storage
from google.cloud.firestore_v1.base_query import FieldFilter

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class FirebaseError(Exception):
    """Raised when Firebase operations fail."""
    pass


class FirebaseManager:
    """
    Firebase Manager for FYP-RoleMarket Web App.

    Handles:
    - Image storage as Base64 in Firebase Storage
    - Design generation records (Firestore collection: 'design_generations')
    - Sales prediction records (Firestore collection: 'sales_predictions')
    """

    # Firestore collection names
    COLLECTION_DESIGNS = 'design_generations'
    COLLECTION_PREDICTIONS = 'sales_predictions'

    # Firebase Storage bucket name (will be set from env)
    STORAGE_BUCKET = None

    def __init__(
        self,
        service_account_key_path: Optional[str] = None,
        storage_bucket: Optional[str] = None
    ):
        """
        Initialize Firebase Manager.

        Args:
            service_account_key_path: Path to Firebase service account key JSON file
                                     (optional, defaults to FIREBASE_SERVICE_ACCOUNT_KEY env var)
            storage_bucket: Firebase Storage bucket name
                           (optional, defaults to FIREBASE_STORAGE_BUCKET env var)

        Raises:
            FirebaseError: If initialization fails
        """
        # Get config from environment
        if service_account_key_path is None:
            service_account_key_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY')

        if storage_bucket is None:
            storage_bucket = os.getenv('FIREBASE_STORAGE_BUCKET')

        if not service_account_key_path:
            raise FirebaseError(
                "Firebase service account key not provided. "
                "Set FIREBASE_SERVICE_ACCOUNT_KEY environment variable."
            )

        if not storage_bucket:
            raise FirebaseError(
                "Firebase storage bucket not provided. "
                "Set FIREBASE_STORAGE_BUCKET environment variable."
            )

        self.STORAGE_BUCKET = storage_bucket

        # Initialize Firebase Admin SDK (only once)
        try:
            if not firebase_admin._apps:
                logger.info("Initializing Firebase Admin SDK...")
                cred = credentials.Certificate(service_account_key_path)
                firebase_admin.initialize_app(cred, {
                    'storageBucket': self.STORAGE_BUCKET
                })
                logger.info(f"✅ Firebase initialized with bucket: {self.STORAGE_BUCKET}")
            else:
                logger.info("Firebase Admin SDK already initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            raise FirebaseError(f"Firebase initialization failed: {str(e)}")

        # Get Firestore client
        self.db = firestore.client()

        # Get Storage bucket
        self.bucket = storage.bucket()

        logger.info("FirebaseManager initialized successfully")

    # ==================== Image Storage ====================

    def upload_image_as_base64(
        self,
        image: Union[Image.Image, str],
        folder: str = "designs",
        filename: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Upload PIL Image or image path to Firebase Storage as Base64.

        Args:
            image: PIL Image object or path to image file
            folder: Storage folder (default: "designs")
            filename: Custom filename (optional, auto-generated if not provided)

        Returns:
            Dictionary containing:
            {
                'storage_path': 'folder/filename.png',
                'download_url': 'https://...',
                'base64_data': 'iVBORw0KGgo...'
            }

        Raises:
            FirebaseError: If upload fails

        Example:
            >>> manager = FirebaseManager()
            >>> result = manager.upload_image_as_base64(
            ...     image=Image.open("design.png"),
            ...     folder="designs",
            ...     filename="lulu_design_001.png"
            ... )
            >>> print(result['download_url'])
        """
        try:
            # Load image
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                raise ValueError("image must be PIL.Image or file path")

            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"image_{timestamp}.png"

            # Convert image to Base64
            logger.info(f"Converting image to Base64: {filename}")
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
            base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Upload to Firebase Storage
            storage_path = f"{folder}/{filename}"
            blob = self.bucket.blob(storage_path)

            # Set content type
            blob.content_type = 'image/png'

            # Upload Base64 data
            logger.info(f"Uploading to Firebase Storage: {storage_path}")
            blob.upload_from_string(
                base64.b64decode(base64_data),
                content_type='image/png'
            )

            # Make blob publicly accessible
            blob.make_public()

            # Get download URL
            download_url = blob.public_url

            logger.info(f"✅ Image uploaded successfully: {download_url}")

            return {
                'storage_path': storage_path,
                'download_url': download_url,
                'base64_data': base64_data
            }

        except Exception as e:
            logger.error(f"Image upload failed: {e}")
            raise FirebaseError(f"Failed to upload image: {str(e)}")

    def download_image_from_base64(self, storage_path: str) -> Image.Image:
        """
        Download image from Firebase Storage and decode from Base64.

        Args:
            storage_path: Firebase Storage path (e.g., "designs/image_001.png")

        Returns:
            PIL Image object

        Raises:
            FirebaseError: If download fails
        """
        try:
            logger.info(f"Downloading image from: {storage_path}")
            blob = self.bucket.blob(storage_path)

            # Download as bytes
            image_bytes = blob.download_as_bytes()

            # Decode from Base64 (if stored as Base64)
            # Note: If uploaded using upload_image_as_base64, it's already decoded
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            logger.info(f"✅ Image downloaded successfully")
            return image

        except Exception as e:
            logger.error(f"Image download failed: {e}")
            raise FirebaseError(f"Failed to download image: {str(e)}")

    # ==================== Design Generation Records CRUD ====================

    def create_design_record(
        self,
        prompt: str,
        keywords: List[str],
        season: str,
        reference_image: str,
        image_url: str,
        image_storage_path: str,
        clip_similarity: float,
        generation_time: float,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Create a new design generation record in Firestore.

        Args:
            prompt: Generated Midjourney prompt
            keywords: List of extracted keywords
            season: Season context (Spring/Summer/Fall/Winter)
            reference_image: Reference image filename
            image_url: Firebase Storage download URL
            image_storage_path: Firebase Storage path
            clip_similarity: CLIP similarity score (0.0 - 1.0)
            generation_time: Generation time in seconds
            metadata: Additional metadata (optional)

        Returns:
            Document ID of created record

        Raises:
            FirebaseError: If creation fails

        Example:
            >>> doc_id = manager.create_design_record(
            ...     prompt="Lulu Pig celebrating Christmas...",
            ...     keywords=["Christmas", "cozy", "warm"],
            ...     season="Winter",
            ...     reference_image="lulu_pig_ref_1.jpg",
            ...     image_url="https://...",
            ...     image_storage_path="designs/image_001.png",
            ...     clip_similarity=0.8534,
            ...     generation_time=12.5
            ... )
        """
        try:
            # Prepare document data
            doc_data = {
                'prompt': prompt,
                'keywords': keywords,
                'season': season,
                'reference_image': reference_image,
                'image_url': image_url,
                'image_storage_path': image_storage_path,
                'clip_similarity': clip_similarity,
                'generation_time': generation_time,
                'created_at': firestore.SERVER_TIMESTAMP,
                'updated_at': firestore.SERVER_TIMESTAMP
            }

            # Add optional metadata
            if metadata:
                doc_data['metadata'] = metadata

            # Create document
            logger.info(f"Creating design record in Firestore...")
            doc_ref = self.db.collection(self.COLLECTION_DESIGNS).document()
            doc_ref.set(doc_data)

            doc_id = doc_ref.id
            logger.info(f"✅ Design record created: {doc_id}")

            return doc_id

        except Exception as e:
            logger.error(f"Failed to create design record: {e}")
            raise FirebaseError(f"Failed to create design record: {str(e)}")

    def get_design_record(self, doc_id: str) -> Optional[Dict]:
        """
        Get a single design generation record by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document data as dictionary, or None if not found

        Raises:
            FirebaseError: If retrieval fails
        """
        try:
            doc_ref = self.db.collection(self.COLLECTION_DESIGNS).document(doc_id)
            doc = doc_ref.get()

            if doc.exists:
                data = doc.to_dict()
                data['id'] = doc.id
                return data
            else:
                logger.warning(f"Design record not found: {doc_id}")
                return None

        except Exception as e:
            logger.error(f"Failed to get design record: {e}")
            raise FirebaseError(f"Failed to get design record: {str(e)}")

    def list_design_records(
        self,
        limit: int = 50,
        order_by: str = 'created_at',
        descending: bool = True,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        List design generation records with optional filtering.

        Args:
            limit: Maximum number of records to return (default: 50)
            order_by: Field to order by (default: 'created_at')
            descending: Order descending (default: True)
            filters: Optional filters {'field': 'value'}
                    Example: {'season': 'Winter', 'clip_similarity': ('>', 0.8)}

        Returns:
            List of document dictionaries

        Raises:
            FirebaseError: If listing fails

        Example:
            >>> # Get top 10 designs with high CLIP similarity
            >>> records = manager.list_design_records(
            ...     limit=10,
            ...     order_by='clip_similarity',
            ...     descending=True
            ... )
        """
        try:
            query = self.db.collection(self.COLLECTION_DESIGNS)

            # Apply filters
            if filters:
                for field, value in filters.items():
                    if isinstance(value, tuple) and len(value) == 2:
                        # Comparison filter: ('>', 0.8)
                        operator, filter_value = value
                        query = query.where(filter=FieldFilter(field, operator, filter_value))
                    else:
                        # Equality filter
                        query = query.where(filter=FieldFilter(field, '==', value))

            # Order and limit
            if descending:
                query = query.order_by(order_by, direction=firestore.Query.DESCENDING)
            else:
                query = query.order_by(order_by, direction=firestore.Query.ASCENDING)

            query = query.limit(limit)

            # Execute query
            docs = query.stream()

            results = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                results.append(data)

            logger.info(f"✅ Retrieved {len(results)} design records")
            return results

        except Exception as e:
            logger.error(f"Failed to list design records: {e}")
            raise FirebaseError(f"Failed to list design records: {str(e)}")

    def update_design_record(self, doc_id: str, updates: Dict) -> bool:
        """
        Update an existing design generation record.

        Args:
            doc_id: Document ID
            updates: Dictionary of fields to update

        Returns:
            True if successful, False if document not found

        Raises:
            FirebaseError: If update fails

        Example:
            >>> manager.update_design_record(
            ...     doc_id="abc123",
            ...     updates={'clip_similarity': 0.9, 'metadata': {'validated': True}}
            ... )
        """
        try:
            doc_ref = self.db.collection(self.COLLECTION_DESIGNS).document(doc_id)

            # Check if document exists
            if not doc_ref.get().exists:
                logger.warning(f"Design record not found: {doc_id}")
                return False

            # Add updated_at timestamp
            updates['updated_at'] = firestore.SERVER_TIMESTAMP

            # Update document
            doc_ref.update(updates)
            logger.info(f"✅ Design record updated: {doc_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to update design record: {e}")
            raise FirebaseError(f"Failed to update design record: {str(e)}")

    def delete_design_record(self, doc_id: str) -> bool:
        """
        Delete a design generation record.

        Args:
            doc_id: Document ID

        Returns:
            True if successful, False if document not found

        Raises:
            FirebaseError: If deletion fails
        """
        try:
            doc_ref = self.db.collection(self.COLLECTION_DESIGNS).document(doc_id)

            # Check if document exists
            if not doc_ref.get().exists:
                logger.warning(f"Design record not found: {doc_id}")
                return False

            # Delete document
            doc_ref.delete()
            logger.info(f"✅ Design record deleted: {doc_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to delete design record: {e}")
            raise FirebaseError(f"Failed to delete design record: {str(e)}")

    # ==================== Sales Prediction Records CRUD ====================

    def create_prediction_record(
        self,
        design_id: str,
        season: str,
        predicted_sales: float,
        lower_bound: float,
        upper_bound: float,
        confidence: float,
        mae: float,
        trends_history: List[float],
        clip_similarity: float,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Create a new sales prediction record in Firestore.

        Args:
            design_id: Reference to design generation record ID
            season: Target season (Spring/Summer/Fall/Winter)
            predicted_sales: Predicted sales quantity
            lower_bound: Prediction lower bound
            upper_bound: Prediction upper bound
            confidence: Model confidence (R² score)
            mae: Mean Absolute Error
            trends_history: Google Trends history [Q-3, Q-2, Q-1, Q0]
            clip_similarity: CLIP similarity of design
            metadata: Additional metadata (optional)

        Returns:
            Document ID of created record

        Raises:
            FirebaseError: If creation fails

        Example:
            >>> doc_id = manager.create_prediction_record(
            ...     design_id="abc123",
            ...     season="Spring",
            ...     predicted_sales=2850.5,
            ...     lower_bound=2523.24,
            ...     upper_bound=3177.76,
            ...     confidence=0.6788,
            ...     mae=327.26,
            ...     trends_history=[45, 52, 48, 50],
            ...     clip_similarity=0.8534
            ... )
        """
        try:
            # Prepare document data
            doc_data = {
                'design_id': design_id,
                'season': season,
                'predicted_sales': predicted_sales,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'confidence': confidence,
                'mae': mae,
                'trends_history': trends_history,
                'clip_similarity': clip_similarity,
                'created_at': firestore.SERVER_TIMESTAMP,
                'updated_at': firestore.SERVER_TIMESTAMP
            }

            # Add optional metadata
            if metadata:
                doc_data['metadata'] = metadata

            # Create document
            logger.info(f"Creating prediction record in Firestore...")
            doc_ref = self.db.collection(self.COLLECTION_PREDICTIONS).document()
            doc_ref.set(doc_data)

            doc_id = doc_ref.id
            logger.info(f"✅ Prediction record created: {doc_id}")

            return doc_id

        except Exception as e:
            logger.error(f"Failed to create prediction record: {e}")
            raise FirebaseError(f"Failed to create prediction record: {str(e)}")

    def get_prediction_record(self, doc_id: str) -> Optional[Dict]:
        """
        Get a single prediction record by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document data as dictionary, or None if not found

        Raises:
            FirebaseError: If retrieval fails
        """
        try:
            doc_ref = self.db.collection(self.COLLECTION_PREDICTIONS).document(doc_id)
            doc = doc_ref.get()

            if doc.exists:
                data = doc.to_dict()
                data['id'] = doc.id
                return data
            else:
                logger.warning(f"Prediction record not found: {doc_id}")
                return None

        except Exception as e:
            logger.error(f"Failed to get prediction record: {e}")
            raise FirebaseError(f"Failed to get prediction record: {str(e)}")

    def list_prediction_records(
        self,
        design_id: Optional[str] = None,
        limit: int = 50,
        order_by: str = 'created_at',
        descending: bool = True,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        List prediction records with optional filtering.

        Args:
            design_id: Filter by design ID (optional)
            limit: Maximum number of records to return (default: 50)
            order_by: Field to order by (default: 'created_at')
            descending: Order descending (default: True)
            filters: Optional filters {'field': 'value'}
                    Example: {'season': 'Spring', 'predicted_sales': ('>', 3000)}

        Returns:
            List of document dictionaries

        Raises:
            FirebaseError: If listing fails

        Example:
            >>> # Get all predictions for a specific design
            >>> records = manager.list_prediction_records(design_id="abc123")
        """
        try:
            query = self.db.collection(self.COLLECTION_PREDICTIONS)

            # Filter by design_id if provided
            if design_id:
                query = query.where(filter=FieldFilter('design_id', '==', design_id))

            # Apply additional filters
            if filters:
                for field, value in filters.items():
                    if isinstance(value, tuple) and len(value) == 2:
                        # Comparison filter
                        operator, filter_value = value
                        query = query.where(filter=FieldFilter(field, operator, filter_value))
                    else:
                        # Equality filter
                        query = query.where(filter=FieldFilter(field, '==', value))

            # Order and limit
            if descending:
                query = query.order_by(order_by, direction=firestore.Query.DESCENDING)
            else:
                query = query.order_by(order_by, direction=firestore.Query.ASCENDING)

            query = query.limit(limit)

            # Execute query
            docs = query.stream()

            results = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                results.append(data)

            logger.info(f"✅ Retrieved {len(results)} prediction records")
            return results

        except Exception as e:
            logger.error(f"Failed to list prediction records: {e}")
            raise FirebaseError(f"Failed to list prediction records: {str(e)}")

    def update_prediction_record(self, doc_id: str, updates: Dict) -> bool:
        """
        Update an existing prediction record.

        Args:
            doc_id: Document ID
            updates: Dictionary of fields to update

        Returns:
            True if successful, False if document not found

        Raises:
            FirebaseError: If update fails
        """
        try:
            doc_ref = self.db.collection(self.COLLECTION_PREDICTIONS).document(doc_id)

            # Check if document exists
            if not doc_ref.get().exists:
                logger.warning(f"Prediction record not found: {doc_id}")
                return False

            # Add updated_at timestamp
            updates['updated_at'] = firestore.SERVER_TIMESTAMP

            # Update document
            doc_ref.update(updates)
            logger.info(f"✅ Prediction record updated: {doc_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to update prediction record: {e}")
            raise FirebaseError(f"Failed to update prediction record: {str(e)}")

    def delete_prediction_record(self, doc_id: str) -> bool:
        """
        Delete a prediction record.

        Args:
            doc_id: Document ID

        Returns:
            True if successful, False if document not found

        Raises:
            FirebaseError: If deletion fails
        """
        try:
            doc_ref = self.db.collection(self.COLLECTION_PREDICTIONS).document(doc_id)

            # Check if document exists
            if not doc_ref.get().exists:
                logger.warning(f"Prediction record not found: {doc_id}")
                return False

            # Delete document
            doc_ref.delete()
            logger.info(f"✅ Prediction record deleted: {doc_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to delete prediction record: {e}")
            raise FirebaseError(f"Failed to delete prediction record: {str(e)}")
