"""
Pydantic Schemas for Diamond Ring Visual Search API
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DiamondSpecs(BaseModel):
    shape: Optional[str] = Field(None, description="Diamond shape (e.g. Round, Oval, Princess, Emerald)")
    carat: Optional[float] = Field(None, description="Carat weight")
    dimension: Optional[str] = Field(None, description="Stone dimensions in mm")
    type: Optional[str] = Field("Diamond", description="Stone type")


class SideDiamondSpecs(BaseModel):
    qty: Optional[int] = Field(0, description="Total number of side accent diamonds")
    carat: Optional[float] = Field(0.0, description="Total side diamond carat weight")
    color: Optional[str] = Field("G-H", description="Diamond color/clarity grade")


class ColorVariant(BaseModel):
    style_number: str
    metal_color: str
    band_color: str
    metal_category: str
    prong_style: str
    prong_color: str
    prong_summary: str
    primary_image: str
    video_url: Optional[str] = None


class VisualSearchResultItem(BaseModel):
    rank: int = Field(..., description="Relevance rank (1 = Best Match)")
    similarity_score: float = Field(..., description="Visual match confidence percentage (0-100%)")
    product_id: str = Field(..., description="Unique product ID")
    style_number: str = Field(..., description="Style number / SKU")
    base_sku: str = Field(..., description="Base model SKU")
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field("", description="Product description")
    vendor: str = Field("Overnight Mountings", description="Vendor / Manufacturer")
    matched_image_url: str = Field(..., description="URL of the exact matching ring view")

    # Core metadata
    shape: str = Field(..., description="Center diamond shape")
    color: str = Field(..., description="Band metal color / diamond color")
    band_type: str = Field(..., description="Band type")
    band_color: str = Field(..., description="Band metal color")
    band_architecture: str = Field(..., description="Band shank architecture")
    prong_color: str = Field(..., description="Prong metal color")
    prong_style: str = Field(..., description="Prong setting style")
    prong_summary: str = Field(..., description="Complete prong & setting summary")
    style: str = Field(..., description="Overall ring design aesthetic")

    # Additional specs
    center_diamond: Optional[DiamondSpecs] = None
    side_diamonds: Optional[SideDiamondSpecs] = None
    all_images: List[str] = Field(default_factory=list)
    videos: List[str] = Field(default_factory=list)
    default_video: Optional[str] = None
    color_variants: List[ColorVariant] = Field(default_factory=list)
    finger_size: Optional[float] = 7.0
    polish_weight_grams: Optional[float] = 2.5


class QueryAnalysis(BaseModel):
    detected_shape: str = Field(..., description="AI-predicted center diamond cut shape")
    detected_band_color: str = Field(..., description="AI-predicted band metal color")
    detected_band_architecture: str = Field(..., description="AI-predicted band shank architecture")
    detected_prong_style: str = Field(..., description="AI-predicted prong setting style")
    detected_style: str = Field(..., description="AI-predicted overall ring aesthetic style")
    color_probabilities: Dict[str, Any] = Field(default_factory=dict)


class VisualSearchResponse(BaseModel):
    success: bool = True
    query_analysis: QueryAnalysis
    total_matches: int
    results: List[Dict[str, Any]]


class AttributeTaxonomyResponse(BaseModel):
    shapes: List[str]
    band_colors: List[str]
    band_types: List[str]
    band_architectures: List[str]
    prong_styles: List[str]
    styles: List[str]
    total_products: int
    total_search_images: int


class ImageUrlSearchRequest(BaseModel):
    image_url: str = Field(..., description="Direct image URL of the diamond ring to search")
    top_k: Optional[int] = Field(12, ge=1, le=50, description="Number of top matches to return")
    shape: Optional[str] = Field(None, description="Optional filter by diamond shape")
    band_color: Optional[str] = Field(None, description="Optional filter by band metal color")
    band_type: Optional[str] = Field(None, description="Optional filter by band type")
    band_architecture: Optional[str] = Field(None, description="Optional filter by band architecture")
    prong_style: Optional[str] = Field(None, description="Optional filter by prong setting style")
    style: Optional[str] = Field(None, description="Optional filter by ring style")
    crop_x1: Optional[float] = Field(None, description="Crop left coordinate")
    crop_y1: Optional[float] = Field(None, description="Crop top coordinate")
    crop_x2: Optional[float] = Field(None, description="Crop right coordinate")
    crop_y2: Optional[float] = Field(None, description="Crop bottom coordinate")


class Base64SearchRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image string")
    top_k: Optional[int] = Field(12, ge=1, le=50)
    shape: Optional[str] = None
    band_color: Optional[str] = None
    band_type: Optional[str] = None
    band_architecture: Optional[str] = None
    prong_style: Optional[str] = None
    style: Optional[str] = None


class ImageUrlMetadataRequest(BaseModel):
    image_url: str = Field(..., description="Direct image URL or base64 data URL of the diamond ring to extract metadata from")
    crop_x1: Optional[float] = Field(None, description="Optional crop left coordinate (pixels or 0.0-1.0 relative)")
    crop_y1: Optional[float] = Field(None, description="Optional crop top coordinate (pixels or 0.0-1.0 relative)")
    crop_x2: Optional[float] = Field(None, description="Optional crop right coordinate (pixels or 0.0-1.0 relative)")
    crop_y2: Optional[float] = Field(None, description="Optional crop bottom coordinate (pixels or 0.0-1.0 relative)")


class ExtractedRingMetadata(BaseModel):
    diamond_shape: str = Field(..., description="Detected diamond shape cut (Round, Oval, Cushion, Princess, Emerald, Pear, etc.)")
    shape: str = Field(..., description="Alias for diamond shape")
    band_metal_tone: str = Field(..., description="Detected band metal tone (14K Yellow Gold, 14K White Gold, 14K Rose Gold, Platinum)")
    band_color: str = Field(..., description="Alias for band metal tone")
    color: str = Field(..., description="Alias for color/metal")
    band_architecture: str = Field(..., description="Detected band shank architecture (Classic Straight Shank, Split Shank, Twisted Shank, etc.)")
    band_configuration: str = Field(..., description="Alias for band architecture")
    band_type: str = Field("Plain Solitaire Band", description="Band type classification")
    prong_setting: str = Field(..., description="Detected prong setting style (Classic 4-Prong Setting, 6-Prong Setting, Bezel Setting, etc.)")
    prong_style: str = Field(..., description="Alias for prong setting")
    prong_color: str = Field(..., description="Prong metal color")
    ring_style: str = Field(..., description="Detected ring aesthetic style (Solitaire, Halo, Three-Stone, Modern Classic, Vintage)")
    style: str = Field(..., description="Alias for ring style")
    confidence_scores: Dict[str, float] = Field(default_factory=dict, description="Confidence score per attribute")
    all_probabilities: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Detailed category probability distributions")


class ImageMetadataResponse(BaseModel):
    success: bool = True
    metadata: ExtractedRingMetadata
    uploaded_image_metadata: ExtractedRingMetadata
    query_analysis: Dict[str, Any]
    region_detection: Dict[str, Any]
    image_dimensions: Dict[str, int]
    inference_time_ms: float

