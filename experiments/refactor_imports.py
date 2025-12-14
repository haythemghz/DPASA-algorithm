
import os
import glob

def refactor_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = content
        # Replace imports
        new_content = new_content.replace('from sdm ', 'from dpasa ')
        new_content = new_content.replace('from sdm.', 'from dpasa.')
        new_content = new_content.replace('import sdm\n', 'import dpasa\n')
        
        # Replace class names if old ones linger (though I verified most)
        new_content = new_content.replace('SDMOptimizer', 'DPASAOptimizer')
        new_content = new_content.replace('SDM optimizer', 'DPASA optimizer')
        new_content = new_content.replace('SDM algorithm', 'DPASA algorithm')
        
        # Specific fix for imports observed in grep
        new_content = new_content.replace('from sdm.algorithm', 'from dpasa.optimizer') 

        if content != new_content:
            print(f"Updating {filepath}")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

root_dir = r"c:\Users\Dell\Desktop\The_Religion_Inspired_Metaheuristic__A_Sociocultural_Framework_for_Global_Optimization\SDM-algorithm"

for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.py') or file.endswith('.md'):
            refactor_file(os.path.join(subdir, file))
