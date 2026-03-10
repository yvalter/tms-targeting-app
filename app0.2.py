from sympy import symbols, Eq, nsolve, sqrt, pi
from flask import Flask, request, render_template, jsonify, send_file, g
from celery import Celery 
import nibabel as nib
import numpy as np
import ants
import pandas as pd
from stl import mesh
import sys
import trimesh
import trimesh.smoothing
import math
import traceback
import logging
import subprocess
import tempfile
import os
import meshio
from whitenoise import WhiteNoise
from pathlib import Path
from tqdm import tqdm
import uuid
import time
import shutil
from werkzeug.utils import secure_filename
import scipy.spatial.transform as sst

FSL_DIR = os.environ.get('FSL_DIR', '/Users/Soterixmedical/fsl')
BET_BIN = os.path.join(FSL_DIR, 'bin', 'bet')

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0' # Ensure Redis is running
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
ALLOWED_EXTENSIONS = {'nii', 'nii.gz'}

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and \
           any(filename.lower().endswith('.' + ext) for ext in ALLOWED_EXTENSIONS)
EPS = 1e-9  
        
def resolve_mesh_path() -> str:
    """Return path to scaled mesh, or None if not found."""
    scaled = Path("static/scaled_head.stl")
    return str(scaled) if scaled.exists() else None

def load_and_prep_mesh(path_str: str):
    mesh = trimesh.load(path_str, process=False)
    cent, norms = mesh.triangles_center, mesh.face_normals
    mesh.update_faces((norms * (cent - mesh.centroid)).sum(1) > 0)
    mesh.remove_unreferenced_vertices()
    return mesh

def _as_vec3(x):
    """Convert input to 3D vector, return None if invalid."""
    try:
        a = np.asarray(x, dtype=float).reshape(-1)
        if a.size >= 3:
            return np.array([a[0], a[1], a[2]], dtype=float)
        return None
    except (ValueError, TypeError):
        return None

def snap_landmarks(mesh, landmarks_dict):
    pq = trimesh.proximity.ProximityQuery(mesh)
    names = list(landmarks_dict.keys())
    raw = np.array(list(landmarks_dict.values()))
    snapped = []

    for i, pt in enumerate(raw):
        snapped_pt = pq.on_surface([pt])[0][0]
        logger.info(f"Snapped {names[i]}: raw={pt}, snapped={snapped_pt}")
        snapped.append(snapped_pt)

    return {n: snapped[i] for i, n in enumerate(names)}, pq

def calculate_abcs(head_circ, tragus_tragus, nasion_inion):
    """Calculate ellipsoid axes with better error handling."""
    try:
        logger.debug(f"Calculating ABCs for: head_circ={head_circ}, tragus={tragus_tragus}, nasion={nasion_inion}")
        
        P1 = head_circ
        P2 = tragus_tragus * 1.4
        P3 = nasion_inion * 1.6
        
        a, b, c = symbols('a b c', real=True, positive=True)
        
        eq1 = Eq(
            P1,
            pi * (a + b) * (1 + (3 * ((a - b) ** 2 / (a + b) ** 2)) / (10 + sqrt(4 - 3 * ((a - b) ** 2 / (a + b) ** 2))))
        )
        eq2 = Eq(
            P2,
            pi * (a + c) * (1 + (3 * ((a - c) ** 2 / (a + c) ** 2)) / (10 + sqrt(4 - 3 * ((a - c) ** 2 / (a + c) ** 2))))
        )
        eq3 = Eq(
            P3,
            pi * (b + c) * (1 + (3 * ((b - c) ** 2 / (b + c) ** 2)) / (10 + sqrt(4 - 3 * ((b - c) ** 2 / (b + c) ** 2))))
        )
        
        # Try different initial guesses if the first one fails
        initial_guesses = [(10, 10, 10), (5, 8, 7), (15, 12, 8), (8, 6, 5)]
        
        for guess in initial_guesses:
            try:
                solution = nsolve([eq1, eq2, eq3], [a, b, c], guess)
                result = [float(val.evalf(3)) for val in solution]
                logger.debug(f"Successfully calculated ABCs: {result}")
                return result
            except Exception as e:
                logger.debug(f"Failed with initial guess {guess}: {e}")
                continue
        
        raise ValueError("Could not solve ellipsoid equations with any initial guess")
        
    except Exception as e:
        logger.error(f"Error in calculate_abcs: {e}")
        raise

def calculate_f3(dx, dy, circum):
    """Calculate F3 coordinates with input validation."""
    try:
        logger.debug(f"Calculating F3 for: dx={dx}, dy={dy}, circum={circum}")
        
        # Input validation
        if any(x <= 0 for x in [dx, dy, circum]):
            raise ValueError("All measurements must be positive")
            
        r1 = dx * 0.4
        r2 = dy * 0.4
        
        # Check for division by zero
        if r1 == 0:
            raise ValueError("dx cannot be zero")
            
        mab = ((r1 * -0.58779) + (r2 / 2)) / r1
        
        denominator = (r2 * 0.4694716) - (r1 / 2)
        if abs(denominator) < 1e-10:
            raise ValueError("Invalid measurement combination causing division by zero")
            
        mcd = (r2 * -0.8829476) / denominator
        
        if abs(mab - mcd) < 1e-10:
            raise ValueError("Invalid measurement combination")
            
        xf = (r2 - (mcd * r1)) / (2 * (mab - mcd))
        yf = (mab * xf) - (r2 / 2)
        rf = math.hypot(xf, yf) * 0.91
        ang = math.degrees(math.atan2(yf, xf)) + 90
        circdist = (ang / 90) * (circum / 4)
        
        result = {
            'circumferential_dist': circdist,
            'vertex_dist': rf,
            'vertex_dist_adjusted': rf + 0.35,
            'angle': ang
        }
        
        logger.debug(f"F3 calculation result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in calculate_f3: {e}")
        raise
def read_ants_affine_matrix(mat_file):
    """Read ANTs affine transform and convert to 4x4 matrix using the ants library"""
    import numpy as np
    import ants
    
    # Use the ants library to read the binary transform file correctly
    tx = ants.read_transform(mat_file)
    
    # Extract the parameters (rotation/scale/shear and translation)
    # Parameters for an affine transform are typically a 12-element array
    params = tx.parameters
    
    # Extract fixed parameters (the center of rotation)
    fixed_params = tx.fixed_parameters
    
    # Initialize a 4x4 identity matrix
    affine_matrix = np.eye(4)
    
    # The first 9 parameters represent the 3x3 linear transformation matrix
    # They are stored in row-major order
    affine_matrix[:3, :3] = params[:9].reshape(3, 3)
    
    # The last 3 parameters represent the translation component
    affine_matrix[:3, 3] = params[9:12]
    
    # Adjust the translation for the center of rotation (FixedParameters)
    # Formula: T_adjusted = Translation + Center - (Matrix * Center)
    center = fixed_params[:3]
    affine_matrix[:3, 3] += center - (affine_matrix[:3, :3] @ center)
    
    return affine_matrix

def convert_ants_lps_to_ras(affine_lps):
    """Convert an ANTs affine matrix from LPS to RAS coordinate convention."""
    lps_to_ras = np.diag([-1, -1, 1, 1]).astype(float)
    return lps_to_ras @ affine_lps @ lps_to_ras

def log_rotation_from_matrix(name, matrix):
    """Log euler angles extracted from a 4x4 affine matrix for debugging."""
    import scipy.spatial.transform as sst
    R = matrix[:3, :3]
    scales = np.linalg.norm(R, axis=0)
    if np.any(scales < 1e-6):
        logger.warning(f"{name}: degenerate scale, cannot extract rotation")
        return
    R_pure = R / scales
    U, _, Vt = np.linalg.svd(R_pure)
    R_ortho = U @ Vt
    r = sst.Rotation.from_matrix(R_ortho)
    euler = r.as_euler('xyz', degrees=True)
    logger.info(f"{name} rotation (euler xyz degrees): roll={euler[0]:.2f}, pitch={euler[1]:.2f}, yaw={euler[2]:.2f}")
    logger.info(f"{name} translation: {matrix[:3, 3]}")

def apply_transform_to_stl(stl_path, affine_matrix, output_path):
    """Apply 4x4 affine transformation to STL file"""
    import numpy as np
    from stl import mesh
    
    # Load the STL file
    stl_mesh = mesh.Mesh.from_file(stl_path)
    
    # Apply transformation to each vertex
    for i in range(len(stl_mesh.vectors)):
        for j in range(3):  # 3 vertices per triangle
            vertex = np.append(stl_mesh.vectors[i][j], 1)  # Homogeneous coordinates
            transformed = affine_matrix @ vertex
            stl_mesh.vectors[i][j] = transformed[:3]
    
    # Recalculate normals
    stl_mesh.update_normals()
    
    # Save transformed STL
    stl_mesh.save(output_path)

def cleanup_session_files(session_id, output_dir):
    """Remove all session-specific files after they've been served to the client."""
    patterns = [
        os.path.join(output_dir, f"{session_id}_skin.stl"),
        os.path.join(output_dir, f"{session_id}_brain.stl"),
        os.path.join(output_dir, f"{session_id}_target_DLPFC.stl"),
        os.path.join(tempfile.gettempdir(), f"{session_id}_brain.nii.gz"),
        os.path.join(tempfile.gettempdir(), f"input_{session_id}.nii"),
        os.path.join(tempfile.gettempdir(), f"input_{session_id}.nii.gz"),
    ]
    for path in patterns:
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Cleaned up: {path}")
        except Exception as e:
            logger.warning(f"Could not remove {path}: {e}")

    # Clean up any ANTs transform files left in temp
    temp_dir = tempfile.gettempdir()
    for fname in os.listdir(temp_dir):
        if session_id in fname or (fname.endswith('.mat') and fname.startswith('tmp')):
            try:
                os.remove(os.path.join(temp_dir, fname))
                logger.info(f"Cleaned up ANTs temp file: {fname}")
            except Exception as e:
                logger.warning(f"Could not remove ANTs temp file {fname}: {e}")
                
def scale_stl(input_path, output_path, scale_matrix, session_id=None):
    """Scale STL file and save to output path. Returns error response if failed."""
    try:
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Input STL not found: {input_path}")
            
        mesh = trimesh.load(input_path)
        vertices = mesh.vertices
        scaled_vertices = np.dot(vertices, scale_matrix)
        mesh.vertices = scaled_vertices
        mesh.export(output_path)
        logger.info(f"STL saved to {output_path}")
        return None  # Success
    except Exception as e:
        logger.error(f"STL scaling error: {str(e)}")
        return jsonify({'error': f"STL scaling failed: {str(e)}"}), 500
def scale_brain_stl(input_path, output_path, scale_matrix, session_id=None):
    """Scale brain STL file and save to output path. Returns error response if failed."""
    try:
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Input brain STL not found: {input_path}")
            
        mesh = trimesh.load(input_path)
        vertices = mesh.vertices
        scaled_vertices = np.dot(vertices, scale_matrix)
        mesh.vertices = scaled_vertices
        mesh.export(output_path)
        logger.info(f"Brain STL saved to {output_path}")
        return None  # Success
    except Exception as e:
        logger.error(f"Brain STL scaling error: {str(e)}")
        return jsonify({'error': f"Brain STL scaling failed: {str(e)}"}), 500
    
def scale_nifti(input_path, output_path, scale_matrix):
    """Scale NIfTI file and save to output path."""
    try:
        logger.info(f"Attempting to scale NIfTI from {input_path} to {output_path}")
        if not Path(input_path).exists():
            logger.error(f"Input NIfTI not found: {input_path}")
            raise FileNotFoundError(f"Input NIfTI not found: {input_path}")
        
        logger.info("Loading NIfTI file...")
        # Load the NIfTI file
        nifti_img = nib.load(input_path)
        data = nifti_img.get_fdata()
        affine = nifti_img.affine.copy()
        
        logger.info(f"Original affine matrix:\n{affine}")
        
        # Apply scaling to the affine matrix
        affine[:3, :3] = np.dot(affine[:3, :3], scale_matrix)
        
        logger.info(f"Scaled affine matrix:\n{affine}")
        
        # Create new NIfTI image with scaled affine
        scaled_img = nib.Nifti1Image(data, affine, nifti_img.header)
        
        # Save the scaled NIfTI
        nib.save(scaled_img, output_path)
        logger.info(f"Scaled NIfTI saved to {output_path}")

        if Path(output_path).exists():
            logger.info(f"Verified: NIfTI file exists at {output_path}")
        else:
            logger.error(f"ERROR: NIfTI file was NOT created at {output_path}")
        
        return None
    except Exception as e:
        logger.error(f"NIfTI scaling error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f"NIfTI scaling failed: {str(e)}"}), 500
        
def log_rotation_from_matrix(name, matrix):
    R = matrix[:3, :3]
    # Remove scale from rotation matrix
    scales = np.linalg.norm(R, axis=0)
    R_pure = R / scales
    r = sst.Rotation.from_matrix(R_pure)
    euler = r.as_euler('xyz', degrees=True)
    logger.info(f"{name} rotation (euler xyz degrees): roll={euler[0]:.2f}, pitch={euler[1]:.2f}, yaw={euler[2]:.2f}")
    logger.info(f"{name} translation: {matrix[:3, 3]}")

def map_mni_to_subject_low_ram(subject_nifti_path, mni_coords_list):
    """
    Performs Low-Res Nonlinear Registration to map MNI coordinates to Subject space.
    """
    logger.info("Starting Low-RAM SyN Mapping...")

    # 1. Load Images
    fixed = ants.image_read(subject_nifti_path)
    moving = ants.image_read("static/MNI152_T1_2mm_brain.nii.gz")

    # 2. DOWNSAMPLE (To keep RAM < 1GB)
    resample_params = (4, 4, 4) 
    fixed_low = ants.resample_image(fixed, resample_params, use_voxels=False, interp_type=0)
    moving_low = ants.resample_image(moving, resample_params, use_voxels=False, interp_type=0)

    # 3. Register
    logger.info("Running SyN registration on low-res proxies...")
    reg = ants.registration(
        fixed=fixed_low,
        moving=moving_low,
        type_of_transform='antsRegistrationSyNQuick[s]',
        reg_iterations=(40, 20, 0),
        verbose=False
    )

    # 4. Flip X and Y signs for ANTs compatibility (ANTs/ITK uses LPS, MNI is RAS)
    pts_data = np.array(mni_coords_list, dtype=float)
    pts_data[:, 0] *= -1
    pts_data[:, 1] *= -1

    # Add a dummy point to prevent ANTsPy from flattening the output
    # This prevents the "Length of values (3) does not match length of index (1)" error
    pts_data_with_dummy = np.vstack([pts_data, [0, 0, 0]])
    pts_df = pd.DataFrame(pts_data_with_dummy, columns=['x', 'y', 'z'])

    # 5. Apply Transform
    warped_df = ants.apply_transforms_to_points(
        dim=3,
        points=pts_df,
        transformlist=reg['fwdtransforms']
    )
    for transform_file in reg['fwdtransforms'] + reg.get('invtransforms', []):
        try:
            if os.path.exists(transform_file):
                os.remove(transform_file)
                logger.info(f"Cleaned up SyN transform: {transform_file}")
        except Exception as e:
            logger.warning(f"Could not remove SyN transform {transform_file}: {e}")

    # 6. Extract result and flip X/Y back to RAS
    subject_coords = warped_df.iloc[:len(mni_coords_list)].copy() # Remove dummy
    subject_coords['x'] *= -1
    subject_coords['y'] *= -1
    
    final_coords = subject_coords[['x', 'y', 'z']].values.tolist()
    logger.info(f"Mapped MNI {mni_coords_list} -> Subject {final_coords}")
    return final_coords

def call_meshtomeasure(fpz, oz, dlpfc, scaled_stl_path, session_id=None):
    try:
        logger.info("Calling meshtomeasure script for path calculations")
        safe_stl_path=Path(scaled_stl_path).as_posix()
        # Create a temporary Python script that imports and runs meshtomeasure
        script_content = f"""
import sys
import numpy as np
sys.path.append('.')

# Set the landmark variables that meshtomeasure will read
FPZ = {list(fpz)}
OZ = {list(oz)}
DLPFC = {list(dlpfc)}

# Copy the scaled STL to the expected location
import shutil
from pathlib import Path
scaled_stl = Path('{safe_stl_path}') # Use the safe path
if scaled_stl.exists():
    shutil.copy(str(scaled_stl), 'SCALED_HEAD.stl')

# Import and run meshtomeasure with JSON output
import meshtomeasure_YK_v0
meshtomeasure_YK_v0.main(output_format='json', silent_progress=True)
"""
        
        # Write the script to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(script_content)
            temp_script_path = temp_file.name
        
        try:
            # Run the script and capture output
            result = subprocess.run(
                [sys.executable, temp_script_path],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=os.getcwd()
            )
            
            if result.returncode != 0:
                raise Exception(f"meshtomeasure script failed with return code {result.returncode}: {result.stderr}")
            # Parse JSON output
            try:
                import json
                # Get the last line of stdout which should contain the JSON
                output_lines = [line.strip() for line in result.stdout.split('\n') if line.strip()]
                if not output_lines:
                    raise Exception("No output received from meshtomeasure")
                
                json_output = output_lines[-1]  # Last non-empty line should be JSON
                results = json.loads(json_output)
                
                # Check for errors in the JSON output
                if 'error' in results:
                    raise Exception(f"meshtomeasure reported error: {results['error']}")
                
                # Validate required fields
                required_fields = ['vertical_length', 'horizontal_length', 'vertical_path', 'horizontal_path']
                for field in required_fields:
                    if field not in results:
                        raise Exception(f"Missing required field '{field}' in meshtomeasure output")
                logger.info(f"Successfully extracted results: vertical={results['vertical_length']}, horizontal={results['horizontal_length']}")
                return results
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON output: {e}")
                logger.error(f"Raw output: {result.stdout}")
                raise Exception("Failed to parse meshtomeasure JSON output")
            
        finally:
            # Clean up temporary files
            try:
                os.unlink(temp_script_path)
            except OSError:
                pass
            
            try:
                os.unlink('SCALED_HEAD.stl')
            except OSError:
                pass
                
    except Exception as e:
        logger.error(f"Error calling meshtomeasure: {e}")
        raise

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/scale', methods=['POST'])
def scale_route():
    try:
        session_id = str(uuid.uuid4())

        logger.info("Processing scale request")
        logger.debug(f"Form data: {dict(request.form)}")
        
        # Parse and validate inputs
        try:
            head_circ = float(request.form['head_circ'])
            tragus = float(request.form['tragus_tragus'])
            nasion = float(request.form['nasion_inion'])
        except KeyError as e:
            logger.error(f"Missing required form field: {e}")
            return jsonify({'error': f"Missing required field: {str(e)}"}), 400
        except ValueError as e:
            logger.error(f"Invalid number format: {e}")
            return jsonify({'error': "All measurements must be valid numbers."}), 400
        
        logger.info(f"Parsed measurements: head_circ={head_circ}, tragus={tragus}, nasion={nasion}")
        
        if not all(x > 0 for x in [head_circ, tragus, nasion]):
            logger.error("Non-positive measurements provided")
            return jsonify({'error': "Measurements must be positive numbers."}), 400
        # Calculate ellipsoid axes
        a, b, c = calculate_abcs(head_circ, tragus, nasion)
        logger.info(f"Calculated ellipsoid axes: a={a}, b={b}, c={c}")

        # Always compute Beam-F3
        f3 = calculate_f3(tragus, nasion, head_circ)
        logger.info(f"Calculated F3: {f3}")

        result_type = request.form.get("result_type", "f3")
        logger.info(f"Result type requested: {result_type}")

        if result_type == "f3":
            logger.info("Returning F3 results")
            return jsonify({
                'success': True,
                'result_type': 'f3',
                'circumferential_dist': f3['circumferential_dist'],
                'vertex_dist': f3['vertex_dist'],
                'vertex_dist_adjusted': f3['vertex_dist_adjusted']
            })

        # Valter-MNI method processing
        logger.info("Processing Valter-MNI request")
        sx = (a / 8.409) * 0.95
        sy = b / 10.31
        sz = c / 9.8157
        scale_matrix = np.array([
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, sz]
        ], dtype=float)
        logger.debug(f"Scale matrix: {scale_matrix}")

        translation_vector = np.array([0, 15.9, -2.56], dtype=float)
        logger.debug(f"Translation vector: {translation_vector}")

        rotation_matrix = np.array([
            [1, 0, 0],
            [0, 1, -0.04],
            [0, 0.04, 1]
        ], dtype=float)
        logger.debug(f"Rotation matrix: {rotation_matrix}")
        
        # Parse target choice and handle custom coordinates
        target_choice = request.form.get('target_choice', 'default')
        input_point = None
        translated_point = None
        rotated_point = None
        scaled_point = None

        if target_choice == 'custom':
            px = request.form.get('point_x', '').strip()
            py = request.form.get('point_y', '').strip()
            pz = request.form.get('point_z', '').strip()
            
            if not (px and py and pz):
                return jsonify({'error': "All custom coordinates (X, Y, Z) are required."}), 400
            
            try:
                input_point = [float(px), float(py), float(pz)]
                pt = np.array(input_point, dtype=float)
                translated_point = pt + translation_vector
                rotated_point = translated_point @ rotation_matrix
                scaled_pt = rotated_point @ scale_matrix
                scaled_pt[0] *= -1
                scaled_pt[1] *= -1
                scaled_point = scaled_pt.tolist()
                logger.debug(f"Custom point: {input_point} -> {scaled_point}")
            except ValueError:
                return jsonify({'error': "Invalid coordinates for custom point."}), 400
        else:
            input_point = [38.0000, -58.9624, 25.8360]
            pt = np.array(input_point, dtype=float)
            scaled_pt = pt @ scale_matrix
            scaled_point = scaled_pt.tolist()
            logger.debug(f"Default point: {input_point} -> {scaled_point}")

        # Scale predefined anatomical landmarks
        pre_fpz = np.array([-0.33, -103.347, -0.619], dtype=float)
        pre_cz = np.array([-0.33, 1.9828, 94.6484], dtype=float)
        pre_oz = np.array([-0.33, 103.347, -0.619], dtype=float)
        
        FPZ = (scale_matrix @ pre_fpz)
        CZ = (scale_matrix @ pre_cz)
        OZ = (scale_matrix @ pre_oz)
        
        FPZ = FPZ.tolist()
        CZ = CZ.tolist()
        OZ = OZ.tolist()
        logger.debug(f"Scaled landmarks: FPZ={FPZ}, CZ={CZ}, OZ={OZ}")
        
        # Scale the STL model
        input_stl = 'static/model.stl'
        output_stl = 'static/scaled_head.stl'
        input_brain_stl = 'static/mni_brain.stl'
        output_brain_stl = 'static/scaled_brain.stl'
        brain_scale_result = scale_brain_stl(input_brain_stl, output_brain_stl, scale_matrix)       
        if brain_scale_result:
            return brain_scale_result
        if not Path(input_stl).exists():
            logger.error(f"Input STL file not found: {input_stl}")
            return jsonify({'error': f"Input STL file not found: {input_stl}"}), 404

        scale_result = scale_stl(input_stl, output_stl, scale_matrix)
        if scale_result:
            return scale_result

        # Snap landmarks to mesh for accurate positioning
        mesh_path = resolve_mesh_path()
        if not mesh_path:
            logger.error("Scaled STL file not found after processing")
            return jsonify({'error': "Scaled STL file not found after processing."}), 500
                
        logger.info(f"Loading mesh from: {mesh_path}")
        mesh = load_and_prep_mesh(mesh_path)
        
        target_point = scaled_point
        overrides = {
            'FPZ': FPZ, 
            'OZ': OZ, 
            'CZ': CZ, 
            'DLPFC': target_point
        }
        
        logger.debug(f"Landmark overrides: {overrides}")
        
        # Snap landmarks to get accurate surface positions
        landmarks_dict = {
            'Fpz': FPZ,
            'Oz': OZ, 
            'Cz': CZ,
            'dlPFC': target_point
        }
        pos, pq = snap_landmarks(mesh, landmarks_dict)

        # Call meshtomeasure script for path calculations
        try:
            path_results = call_meshtomeasure(pos['Fpz'], pos['Oz'], pos['dlPFC'], mesh_path)
            vertical_length = path_results['vertical_length']
            horizontal_length = path_results['horizontal_length']
            vertical_path = path_results['vertical_path']
            horizontal_path = path_results['horizontal_path']
        except Exception as e:
            logger.error(f"Error in meshtomeasure call: {e}")
            return jsonify({'error': f"Path calculation failed: {str(e)}"}), 500

        # Export final mesh
        mesh.export(output_stl)
        DLPFC_centered = pos['dlPFC']

        logger.info("Rendering Valter-MNI results template")
        logger.info(f"DLPFC snapped: {pos['dlPFC']}")
        logger.info(f"DLPFC_centered (sent to client): {DLPFC_centered}")
        logger.info(f"Mesh centroid: {mesh.centroid}")
        # Render Valter-MNI results
        return jsonify({
            'success': True,
            'result_type': 'valter',
            'session_id': session_id,
            'circumferential_dist': round(horizontal_length / 10, 2),
            'vertex_dist_adjusted': round(vertical_length / 10, 2),
            'DLPFC_snapped': DLPFC_centered.tolist(),
            'scaled_point': scaled_point,
            'vertical_path': vertical_path,
            'horizontal_path': horizontal_path,
            'stl_timestamp': str(int(Path(output_stl).stat().st_mtime)),
            'brain_stl_timestamp': str(int(Path(output_brain_stl).stat().st_mtime)),
            'scale_matrix': scale_matrix.tolist()
        })

    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f"Invalid input: {str(ve)}"}), 400
    except FileNotFoundError as fe:
        logger.error(f"FileNotFoundError: {fe}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f"File not found: {str(fe)}"}), 404
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f"Processing error: {str(e)}"}), 500

@celery.task(bind=True)
def process_mri_segmentation_task(self, nifti_path, output_dir, session_id):
    if not os.path.exists(nifti_path):
        raise FileNotFoundError(f"Input file not found: {nifti_path}")

    try:
        generated_files = {}

        # 1. Load patient image
        raw_img = ants.image_read(nifti_path)

        # 2a. REGISTRATION: Full MNI head template -> patient (for skin/head STL)
        logger.info("Starting ANTsPy affine registration: MNI head -> patient...")
        mni_head = ants.image_read("static/MNI152_T1_2mm.nii.gz")
        resample_params = (4, 4, 4)
        raw_img_low  = ants.resample_image(raw_img,  resample_params, use_voxels=False, interp_type=1)
        mni_head_low = ants.resample_image(mni_head, resample_params, use_voxels=False, interp_type=1)
        reg_head = ants.registration(
            fixed=raw_img_low,
            moving=mni_head_low,
            type_of_transform='Affine'
        )
        del raw_img_low, mni_head_low
        affine_matrix_head = read_ants_affine_matrix(reg_head['fwdtransforms'][0])
        affine_matrix_head = convert_ants_lps_to_ras(affine_matrix_head)
        log_rotation_from_matrix("HEAD post-correction", affine_matrix_head)
        logger.info("Head registration complete.")
        logger.info(f"Affine matrix:\n{affine_matrix_head}")
        
        affine_matrix_brain = affine_matrix_head
        logger.info("Reusing head affine matrix for brain STL")
        for transform_file in reg_head['fwdtransforms'] + reg_head.get('invtransforms', []):
            try:
                if os.path.exists(transform_file):
                    os.remove(transform_file)
                    logger.info(f"Cleaned up transform: {transform_file}")
            except Exception as e:
                logger.warning(f"Could not remove transform file {transform_file}: {e}")
        # Extract and log the scale factors from each matrix
        head_scales = np.linalg.norm(affine_matrix_head[:3, :3], axis=0)
        logger.info(f"Scale factors (per axis): {head_scales}")

        # 3a. Apply HEAD transform to skin/head STL
        head_template_stl = os.path.join('static', 'model.stl')
        if os.path.exists(head_template_stl):
            skin_stl_name = f"{session_id}_skin.stl"
            skin_stl_path = os.path.join(output_dir, skin_stl_name)
            mesh_check = trimesh.load('static/model.stl')
            logger.info(f"Head STL centroid: {mesh_check.centroid}")
            logger.info(f"Head STL bounds min: {mesh_check.bounds[0]}")
            logger.info(f"Head STL bounds max: {mesh_check.bounds[1]}")
            apply_transform_to_stl(head_template_stl, affine_matrix_head, skin_stl_path)
            generated_files['skin_stl'] = skin_stl_name
            logger.info(f"Skin STL written: {skin_stl_path}")
        else:
            logger.warning(f"Head template STL not found: {head_template_stl}")

        # 3b. Apply BRAIN transform to brain STL
        brain_template_stl = os.path.join('static', 'mni_brain.stl')
        if os.path.exists(brain_template_stl):
            brain_stl_name = f"{session_id}_brain.stl"
            brain_stl_path = os.path.join(output_dir, brain_stl_name)
            apply_transform_to_stl(brain_template_stl, affine_matrix_brain, brain_stl_path)
            generated_files['brain_stl'] = brain_stl_name
            logger.info(f"Brain STL written: {brain_stl_path}")
        else:
            logger.warning(f"Brain template STL not found: {brain_template_stl}")
        log_rotation_from_matrix("HEAD", affine_matrix_head)

        # 4. DLPFC TARGETING — apply the same affine used for the STLs
        logger.info("Mapping DLPFC MNI coordinate via affine transform...")
        dlpfc_mni = np.array([-38.0, 44.0, 26.0, 1.0])  # homogeneous RAS mm
        dlpfc_subject = affine_matrix_head @ dlpfc_mni
        tx, ty, tz = dlpfc_subject[:3]
        generated_files['DLPFC_coords'] = [tx, ty, tz]
        logger.info(f"DLPFC subject coords (affine): {tx}, {ty}, {tz}")
            
        # Cleanup
        if os.path.exists(nifti_path):
            os.remove(nifti_path)
            
        if os.path.exists(nifti_path):
            os.remove(nifti_path)

        # Schedule session file cleanup after delay
        import threading
        def delayed_cleanup():
            time.sleep(30)
            cleanup_session_files(session_id, output_dir)
        threading.Thread(target=delayed_cleanup, daemon=True).start()

        return generated_files

    except Exception as e:
        logger.error(f"FATAL ERROR: {str(e)}")
        cleanup_session_files(session_id, output_dir)
        raise Exception(f"Processing failed: {str(e)}")
    
@app.route('/process_mri', methods=['POST'])
# Check if this is an MRI Segmentation request
def process_mri_route():
    if 'mri_file' not in request.files:
        return jsonify({'error': 'No MRI file provided'}), 400
   
    file = request.files['mri_file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
   
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
   
    try:
        session_id = str(uuid.uuid4())
        temp_dir = tempfile.gettempdir()
        static_dir = os.path.join(app.root_path, 'static')
       
        filename = secure_filename(file.filename)
        file_extension = "".join(Path(filename).suffixes)
        
        # Construct the temporary path using the actual extension
        input_path = os.path.join(temp_dir, f"input_{session_id}{file_extension}")
        file.save(input_path)
       
        task = process_mri_segmentation_task.apply_async(
            args=[input_path, static_dir, session_id]
        )
       
        return jsonify({
            'success': True,
            'task_id': task.id,
            'session_id': session_id
        })
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({'error': str(e)}), 500
@app.route('/status/<task_id>')
def taskstatus(task_id):
    task = process_mri_segmentation_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {'state': 'PENDING', 'status': 'Pending...'}
    elif task.state == 'PROCESSING':
        response = {'state': 'PROCESSING', 'status': task.info.get('status', '')}
    elif task.state == 'SUCCESS':
        response = {
            'state': 'SUCCESS', 
            'status': 'Done',
            'result': task.result # This contains the filenames
        }
    else:
        # FAILURE or REVOKED
        response = {'state': task.state, 'status': str(task.info)}
    return jsonify(response)    
@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/download_nifti/<session_id>')
def download_nifti(session_id):
    """Generate, download, and clean up the scaled NIfTI file for a given session."""
    nifti_path = None
    try:
        # Get scale_matrix from request args (passed from frontend)
        scale_matrix_str = request.args.get('scale_matrix')
        if not scale_matrix_str:
            return jsonify({'error': 'Scale matrix not provided'}), 400
        
        # Parse the scale matrix
        import json
        scale_matrix = np.array(json.loads(scale_matrix_str), dtype=float)
        
        # Generate the NIfTI file on-demand
        input_nifti = 'static/mni_unscaled.nii'
        temp_dir = tempfile.gettempdir()
        nifti_path = os.path.join(temp_dir, f'scaled_mni_{session_id}.nii')
        
        logger.info(f"Generating NIfTI file on demand for session {session_id}")
        nifti_scale_result = scale_nifti(input_nifti, nifti_path, scale_matrix)
        
        if nifti_scale_result:
            return nifti_scale_result
        
        if not Path(nifti_path).exists():
            return jsonify({'error': 'NIfTI file generation failed'}), 500
        
        # Send the file
        response = send_file(
            nifti_path,
            as_attachment=True,
            download_name='synthetic_mri.nii',
            mimetype='application/octet-stream'
        )
        
        def delete_file():
            try:
                if Path(nifti_path).exists():
                    os.unlink(nifti_path)
                    logger.info(f"Cleaned up NIfTI file: {nifti_path}")
            except Exception as e:
                logger.error(f"Error cleaning up NIfTI file: {e}")

        # Use Flask's after_request to ensure cleanup
        from flask import g
        if not hasattr(g, 'cleanup_files'):
            g.cleanup_files = []
        g.cleanup_files.append(delete_file)

        return response
    
    except Exception as e:
        logger.error(f"Error downloading NIfTI: {e}")
        # Cleanup on error
        try:
            if nifti_path and Path(nifti_path).exists():
                os.unlink(nifti_path)
        except:
            pass
        return jsonify({'error': str(e)}), 500

@app.teardown_request
def cleanup_files(exception=None):
    if hasattr(g, 'cleanup_files'):
        for cleanup_func in g.cleanup_files:
            try:
                cleanup_func()
            except Exception as e:
                logger.error(f"Error in cleanup: {e}")

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)