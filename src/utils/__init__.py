# Import from modular files
from src.utils.file_utils import (
    is_allowed,
    make_dir,
    delete_previous_files,
    make_temp_folder,
    unique_filename as unique_filenameuni,  # Keep old name for backward compatibility
    create_temp_directory_with_age_limit
)

from src.utils.image_utils import (
    get_safe_filename,
    save_image,
    load_image,
    load_config
)

# Export all functions at the package level for backward compatibility
__all__ = [
    'is_allowed',
    'make_dir',
    'delete_previous_files',
    'make_temp_folder',
    'unique_filenameuni',
    'create_temp_directory_with_age_limit',
    'get_safe_filename',
    'save_image',
    'load_image',
    'load_config'
]
