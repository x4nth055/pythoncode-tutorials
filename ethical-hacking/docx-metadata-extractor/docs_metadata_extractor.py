import docx  # Import the docx library for working with Word documents.
from pprint import pprint  # Import the pprint function for pretty printing.

def extract_metadata(docx_file):
    doc = docx.Document(docx_file)  # Create a Document object from the Word document file.
    core_properties = doc.core_properties  # Get the core properties of the document.

    metadata = {}  # Initialize an empty dictionary to store metadata

    # Extract core properties
    for prop in dir(core_properties):  # Iterate over all properties of the core_properties object.
        if prop.startswith('__'):  # Skip properties starting with double underscores (e.g., __elenent). Not needed
            continue
        value = getattr(core_properties, prop)  # Get the value of the property.
        if callable(value):  # Skip callable properties (methods).
            continue
        if prop == 'created' or prop == 'modified' or prop == 'last_printed':  # Check for datetime properties.
            if value:
                value = value.strftime('%Y-%m-%d %H:%M:%S')  # Convert datetime to string format.
            else:
                value = None
        metadata[prop] = value  # Store the property and its value in the metadata dictionary.

    # Extract custom properties (if available).
    try:
        custom_properties = core_properties.custom_properties  # Get the custom properties (if available).
        if custom_properties:  # Check if custom properties exist.
            metadata['custom_properties'] = {}  # Initialize a dictionary to store custom properties.
            for prop in custom_properties:  # Iterate over custom properties.
                metadata['custom_properties'][prop.name] = prop.value  # Store the custom property name and value.
    except AttributeError:
        # Custom properties not available in this version.
        pass  # Skip custom properties extraction if the attribute is not available.

    return metadata  # Return the metadata dictionary.



docx_path = 'test.docx'  # Path to the Word document file.
metadata = extract_metadata(docx_path)  # Call the extract_metadata function.
pprint(metadata)  # Pretty print the metadata dictionary.