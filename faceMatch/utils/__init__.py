# Import from modular files
from faceMatch.utils.file_utils import (
    is_allowed,
    make_dir,
    delete_previous_files,
    make_temp_folder,
    unique_filename as unique_filenameuni,  # Keep old name for backward compatibility
    create_temp_directory_with_age_limit
)

from faceMatch.utils.image_utils import (
    load_image,
    load_config
)

from faceMatch.utils.reference_utils import (
    save_reference_image,
    get_reference_image_path,
    list_reference_images,
    delete_reference_image
)

# Export all functions at the package level for backward compatibility
__all__ = [
    'is_allowed',
    'make_dir',
    'delete_previous_files',
    'make_temp_folder',
    'unique_filenameuni',
    'create_temp_directory_with_age_limit',
    'load_image',
    'load_config',
    'save_reference_image',
    'get_reference_image_path',
    'list_reference_images',
    'delete_reference_image'
]
