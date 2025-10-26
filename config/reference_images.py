# Reference Images Configuration

# Discord CDN URLs (Current - for development/testing)
# Note: These URLs include signature parameters and may expire
CREF_URLS_DISCORD = [
    "https://cdn.discordapp.com/attachments/569198875265728545/1431958293630419134/lulu_pig_ref_1.png?ex=68ff4e95&is=68fdfd15&hm=fe678efd2397311cdf5e06da2e2a86fe60399cc39b2c3ba03c2f6160664f42a7&",
    "https://cdn.discordapp.com/attachments/569198875265728545/1431958294578073730/lulu_pig_ref_2.png?ex=68ff4e95&is=68fdfd15&hm=3c7cb6b68e03c10b1d2f0660f4742bde91cd14161791b146c618defb358d3810&"
]

# Cloud Storage URLs (To be updated after cloud deployment)
# Replace with your actual cloud storage URLs (S3, GCS, Azure Blob, etc.)
CREF_URLS_CLOUD = [
    "https://your-cloud-storage.example.com/reference/lulu_pig_ref_1.png",  # TODO: Update
    "https://your-cloud-storage.example.com/reference/lulu_pig_ref_2.png"   # TODO: Update
]

# Active configuration (switch between Discord and Cloud)
CREF_URLS = CREF_URLS_DISCORD  # Change to CREF_URLS_CLOUD after deployment

# Reference image local paths
REFERENCE_IMAGE_PATHS = [
    "data/reference_images/lulu_pig_ref_1.png",
    "data/reference_images/lulu_pig_ref_2.png"
]

# Character reference settings
CREF_WEIGHT_DEFAULT = 100  # Maximum character consistency (0-100)
CREF_WEIGHT_VARIATIONS = {
    'strict': 100,    # Strong character consistency
    'moderate': 75,   # Good balance
    'flexible': 50    # More creative freedom
}
