"""
Validate all 43 projects for completeness and correctness
"""
import os
import sys

def check_project(project_name):
    """Check if project has all required files"""
    project_path = os.path.join(os.getcwd(), project_name)
    
    if not os.path.exists(project_path):
        return False, "Directory not found"
    
    required_files = ['train.py', 'README.md', 'ARCHITECTURE.md']
    missing = []
    
    for file in required_files:
        if not os.path.exists(os.path.join(project_path, file)):
            missing.append(file)
    
    if missing:
        return False, f"Missing: {', '.join(missing)}"
    
    # Check train.py syntax
    train_py = os.path.join(project_path, 'train.py')
    try:
        with open(train_py, 'r', encoding='utf-8') as f:
            compile(f.read(), train_py, 'exec')
    except SyntaxError as e:
        return False, f"Syntax error in train.py: {e}"
    
    return True, "OK"

def main():
    projects = [f"project_{i:02d}_" for i in range(1, 44)]
    
    print("=" * 80)
    print("VALIDATING ALL 43 PROJECTS")
    print("=" * 80)
    
    valid_count = 0
    invalid_count = 0
    issues = []
    
    for i in range(1, 44):
        # Find project directory
        project_dirs = [d for d in os.listdir('.') if d.startswith(f'project_{i:02d}_')]
        
        if not project_dirs:
            print(f"[{i:02d}] FAIL Project not found")
            invalid_count += 1
            issues.append(f"Project {i:02d}: Not found")
            continue
        
        project_name = project_dirs[0]
        is_valid, message = check_project(project_name)
        
        if is_valid:
            print(f"[{i:02d}] OK {project_name}")
            valid_count += 1
        else:
            print(f"[{i:02d}] FAIL {project_name}: {message}")
            invalid_count += 1
            issues.append(f"Project {i:02d} ({project_name}): {message}")
    
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print(f"\nValid:   {valid_count}/43")
    print(f"Invalid: {invalid_count}/43")
    print(f"Success Rate: {valid_count/43*100:.1f}%")
    
    if issues:
        print("\n" + "=" * 80)
        print("ISSUES FOUND")
        print("=" * 80)
        for issue in issues:
            print(f"  - {issue}")
    
    print("\n" + "=" * 80)
    
    return 0 if invalid_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
