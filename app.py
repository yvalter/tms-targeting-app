from sympy import symbols, Eq, nsolve, sqrt, pi
from flask import Flask, request, render_template, jsonify, send_file, g
import nibabel as nib
import numpy as np
import sys
import trimesh
import math
import traceback
import logging
import subprocess
import tempfile
import os
from whitenoise import WhiteNoise
from pathlib import Path
from tqdm import tqdm
import uuid
import time
import shutil
from pathlib import Path


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

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
    
def call_meshtomeasure(fpz, oz, dlpfc, scaled_stl_path, session_id=None):
    """
    Call the meshtomeasure_YK_v0.py script to calculate path lengths.
    
    Args:
        fpz: List/array of FPZ coordinates [x, y, z]
        oz: List/array of OZ coordinates [x, y, z] 
        dlpfc: List/array of DLPFC coordinates [x, y, z]
        scaled_stl_path: Path to the scaled STL file
        
    Returns:
        dict: Contains vertical_length, horizontal_length, vertical_path, horizontal_path
    """
    try:
        logger.info("Calling meshtomeasure script for path calculations")
        safe_stl_path = scaled_stl_path.replace('\\', '/')
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

# Flask Routes
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
    app.run(host='0.0.0.0', port=port)