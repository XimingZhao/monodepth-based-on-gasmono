import numpy as np
import xml.etree.ElementTree as ET

# 假设的相机内参矩阵和畸变系数
camera_matrix = np.array([[934.5331060961297, 0, 646.162249922997],
                          [0, 934.6172960887559, 358.8661673371651],
                          [0, 0, 1]])
dist_coeffs = np.array([-0.000752406145230268, 0.32913244497159155, 0.0001263833868642026, 0.0007148324123980305, -0.7087706474002451])

def array_to_string(arr):
    return ' '.join(map(str, arr.flatten()))

root = ET.Element("opencv_storage")
ET.SubElement(root, "camera_matrix", type_id="opencv-matrix").text = "\n  3 3 double\n" + array_to_string(camera_matrix) + "\n"
ET.SubElement(root, "distortion_coefficients", type_id="opencv-matrix").text = "\n  1 5 double\n" + array_to_string(dist_coeffs) + "\n"

tree = ET.ElementTree(root)
tree.write("camera_intrinsics.xml")
