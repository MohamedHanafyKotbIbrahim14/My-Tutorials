import streamlit as st
import pandas as pd
import numpy as np
import pulp
import re
from typing import Dict, List, Tuple, Set
import plotly.express as px
import plotly.graph_objects as go
from datetime import time as dt_time

class TutorAssignmentLP:
    """Tutor-Class Assignment using Linear Programming with time conflict detection and degree requirements."""
    
    def __init__(self, classes_df: pd.DataFrame, tutors_df: pd.DataFrame, 
                 preferences: Dict[Tuple[str, str], float],
                 tutor_max_classes: Dict[str, int],
                 tutor_degrees: Dict[str, str],
                 course_diversity_penalty: float = 5.0):
        """
        Initialize the LP problem.
        
        Args:
            classes_df: DataFrame with columns [course, class_id, section, time, mode, course_level]
            tutors_df: DataFrame with tutor information
            preferences: Dict mapping (tutor, course) -> preference_score
            tutor_max_classes: Dict mapping tutor -> max_classes
            tutor_degrees: Dict mapping tutor -> degree (e.g., "PhD", "Master", "Bachelor")
            course_diversity_penalty: Penalty for teaching multiple different courses
        """
        self.classes_df = classes_df
        self.tutors_df = tutors_df
        self.preferences = preferences
        self.tutor_max_classes = tutor_max_classes
        self.tutor_degrees = tutor_degrees
        self.course_diversity_penalty = course_diversity_penalty
        
        self.tutors = list(tutors_df['tutor_name'].unique())
        self.courses = list(classes_df['course'].unique())
        
        self.model = None
        self.x_vars = {}
        self.y_vars = {}  # Binary variable: tutor teaches course
        self.time_conflicts = self._detect_time_conflicts()
        
    def _detect_time_conflicts(self) -> Dict[Tuple[str, str], bool]:
        """
        Detect time conflicts between classes.
        Returns dict mapping (class_id1, class_id2) -> True if they conflict.
        """
        conflicts = {}
        
        # Parse time strings and detect overlaps
        classes_with_times = []
        for idx, row in self.classes_df.iterrows():
            time_str = str(row['time'])
            parsed_times = self._parse_time_string(time_str)
            if parsed_times:
                classes_with_times.append({
                    'class_id': row['class_id'],
                    'times': parsed_times
                })
        
        # Check all pairs for conflicts
        for i, class1 in enumerate(classes_with_times):
            for class2 in classes_with_times[i+1:]:
                if self._times_overlap(class1['times'], class2['times']):
                    conflicts[(class1['class_id'], class2['class_id'])] = True
                    conflicts[(class2['class_id'], class1['class_id'])] = True
        
        return conflicts
    
    def _parse_time_string(self, time_str: str) -> List[Dict]:
        """
        Parse time string like 'Fri 09-10:30(w1-5, 7-10, Col LG01)'
        Returns list of dicts with day, start_time, end_time
        """
        if pd.isna(time_str) or time_str == 'nan':
            return []
        
        time_slots = []
        
        # Pattern to match: Day HH:MM-HH:MM or Day HH-HH:MM
        pattern = r'(Mon|Tue|Wed|Thu|Fri)\s+(\d{1,2})(?::(\d{2}))?-(\d{1,2})(?::(\d{2}))?'
        
        for match in re.finditer(pattern, time_str):
            day = match.group(1)
            start_hour = int(match.group(2))
            start_min = int(match.group(3)) if match.group(3) else 0
            end_hour = int(match.group(4))
            end_min = int(match.group(5)) if match.group(5) else 0
            
            time_slots.append({
                'day': day,
                'start': start_hour * 60 + start_min,  # Convert to minutes
                'end': end_hour * 60 + end_min
            })
        
        return time_slots
    
    def _times_overlap(self, times1: List[Dict], times2: List[Dict]) -> bool:
        """Check if two time slot lists overlap."""
        for t1 in times1:
            for t2 in times2:
                # Same day and overlapping times
                if t1['day'] == t2['day']:
                    if not (t1['end'] <= t2['start'] or t2['end'] <= t1['start']):
                        return True
        return False
    
    def _can_tutor_teach_course(self, tutor: str, course: str, class_row: pd.Series) -> bool:
        """
        Check if tutor can teach this course based on degree and course level.
        - PhD tutors can teach both PG and UG courses
        - Non-PhD tutors can only teach UG courses
        """
        tutor_degree = self.tutor_degrees.get(tutor, "").upper()
        course_level = class_row['course_level']
        
        # PhD can teach both PG and UG
        if 'PHD' in tutor_degree or 'PH.D' in tutor_degree or 'DOCTOR' in tutor_degree:
            return True
        
        # Non-PhD can only teach UG
        if course_level == 'UG':
            return True
        
        # Non-PhD cannot teach PG
        return False
    
    def build_model(self):
        """Build the LP model with course diversity penalty."""
        self.model = pulp.LpProblem("Tutor_Class_Assignment", pulp.LpMaximize)
        
        # Decision variables: x[(tutor, course, class_id)] = 1 if assigned
        self.x_vars = {}
        for tutor in self.tutors:
            for idx, row in self.classes_df.iterrows():
                course = row['course']
                class_id = row['class_id']
                
                # Check if tutor has preference for this course
                has_preference = self.preferences.get((tutor, course), 0) > 0
                
                # Check if tutor can teach based on degree requirements
                can_teach = self._can_tutor_teach_course(tutor, course, row)
                
                # Only create variable if both conditions are met
                if has_preference and can_teach:
                    self.x_vars[(tutor, course, class_id)] = pulp.LpVariable(
                        f"x_{tutor}_{course}_{class_id}",
                        cat='Binary'
                    )
        
        # Binary variables: y[(tutor, course)] = 1 if tutor teaches at least one class from course
        self.y_vars = {}
        for tutor in self.tutors:
            for course in self.courses:
                # Only create y variable if there's at least one possible x variable
                has_possible_assignment = any(
                    (tutor, course, class_id) in self.x_vars
                    for idx, row in self.classes_df.iterrows()
                    if row['course'] == course
                    for class_id in [row['class_id']]
                )
                
                if has_possible_assignment:
                    self.y_vars[(tutor, course)] = pulp.LpVariable(
                        f"y_{tutor}_{course}",
                        cat='Binary'
                    )
        
        # Objective: Maximize preference satisfaction - penalty for teaching multiple courses
        preference_term = pulp.lpSum([
            self.preferences.get((tutor, course), 0) * self.x_vars[(tutor, course, class_id)]
            for (tutor, course, class_id) in self.x_vars.keys()
        ])
        
        # Penalty term: discourage teaching many different courses
        diversity_penalty_term = self.course_diversity_penalty * pulp.lpSum([
            self.y_vars[(tutor, course)]
            for (tutor, course) in self.y_vars.keys()
        ])
        
        self.model += preference_term - diversity_penalty_term
        
        self._add_constraints()
    
    def _add_constraints(self):
        """Add all constraints to the model."""
        
        # Constraint 1: Each class should have exactly ONE tutor (but allow unassigned if infeasible)
        # We use <= instead of == to allow some classes to remain unassigned if necessary
        for idx, row in self.classes_df.iterrows():
            course = row['course']
            class_id = row['class_id']
            
            assigned_tutors = pulp.lpSum([
                self.x_vars[(tutor, course, class_id)]
                for tutor in self.tutors
                if (tutor, course, class_id) in self.x_vars
            ])
            
            # Try to assign exactly 1, but allow 0 if infeasible
            self.model += (
                assigned_tutors <= 1,
                f"Class_Coverage_Max_{course}_{class_id}"
            )
        
        # Constraint 2: Tutor workload limit
        for tutor in self.tutors:
            total_classes = pulp.lpSum([
                self.x_vars[(tutor, course, class_id)]
                for (t, course, class_id) in self.x_vars.keys()
                if t == tutor
            ])
            
            max_classes = self.tutor_max_classes.get(tutor, 3)
            self.model += (
                total_classes <= max_classes,
                f"Tutor_Workload_{tutor}"
            )
        
        # Constraint 3: Time conflict prevention
        for tutor in self.tutors:
            tutor_possible_classes = [
                (course, class_id) 
                for (t, course, class_id) in self.x_vars.keys() 
                if t == tutor
            ]
            
            for i, (course1, class1) in enumerate(tutor_possible_classes):
                for course2, class2 in tutor_possible_classes[i+1:]:
                    if (class1, class2) in self.time_conflicts:
                        self.model += (
                            self.x_vars[(tutor, course1, class1)] + 
                            self.x_vars[(tutor, course2, class2)] <= 1,
                            f"Time_Conflict_{tutor}_{class1}_{class2}"
                        )
        
        # Constraint 4: Link y variables to x variables
        # If tutor teaches any class from a course, y must be 1
        for tutor in self.tutors:
            for course in self.courses:
                if (tutor, course) in self.y_vars:
                    # Get all x variables for this tutor-course combination
                    relevant_x_vars = [
                        self.x_vars[(tutor, course, class_id)]
                        for (t, c, class_id) in self.x_vars.keys()
                        if t == tutor and c == course
                    ]
                    
                    if relevant_x_vars:
                        # y >= x for each class (if any x is 1, y must be 1)
                        for x_var in relevant_x_vars:
                            self.model += (
                                self.y_vars[(tutor, course)] >= x_var,
                                f"Link_Y_X_{tutor}_{course}_{x_var.name}"
                            )
    
    def solve(self):
        """Solve the optimization problem."""
        if self.model is None:
            self.build_model()
        
        solver = pulp.PULP_CBC_CMD(msg=0)
        self.model.solve(solver)
        
        status = pulp.LpStatus[self.model.status]
        
        if status == 'Optimal':
            return self._extract_solution()
        else:
            return {
                'status': status,
                'objective_value': None,
                'assignments': {},
                'tutor_loads': {},
                'unassigned_classes': [],
                'tutor_course_diversity': {}
            }
    
    def _extract_solution(self):
        """Extract solution from solved model."""
        assignments = {}  # (course, class_id) -> tutor
        tutor_loads = {tutor: {'classes': [], 'total': 0, 'courses': set()} for tutor in self.tutors}
        unassigned_classes = []
        tutor_course_diversity = {}
        
        # Extract assignments
        for (tutor, course, class_id), var in self.x_vars.items():
            if var.varValue and var.varValue > 0.5:  # Binary variable
                assignments[(course, class_id)] = tutor
                tutor_loads[tutor]['classes'].append((course, class_id))
                tutor_loads[tutor]['total'] += 1
                tutor_loads[tutor]['courses'].add(course)
        
        # Calculate course diversity for each tutor
        for tutor in self.tutors:
            num_different_courses = len(tutor_loads[tutor]['courses'])
            tutor_course_diversity[tutor] = num_different_courses
        
        # Find unassigned classes
        for idx, row in self.classes_df.iterrows():
            course = row['course']
            class_id = row['class_id']
            if (course, class_id) not in assignments:
                unassigned_classes.append((course, class_id))
        
        return {
            'status': 'Optimal',
            'objective_value': pulp.value(self.model.objective),
            'assignments': assignments,
            'tutor_loads': tutor_loads,
            'unassigned_classes': unassigned_classes,
            'tutor_course_diversity': tutor_course_diversity
        }


def extract_course_codes(text: str) -> List[str]:
    """Extract course codes from preference text."""
    if pd.isna(text) or str(text).strip() == '':
        return []
    
    # Pattern to match course codes like ACTL2102, RISK5001, COMM2501/5501
    pattern = r'\b(ACTL|RISK|COMM)\d{4}(?:/\d{4})?\b'
    codes = [match.group() for match in re.finditer(pattern, str(text))]
    
    return list(set(codes))


def extract_degree(text: str) -> str:
    """
    Extract degree information from text.
    Returns the full text for degree information.
    """
    if pd.isna(text) or str(text).strip() == '':
        return "Not Specified"
    
    return str(text).strip()


def normalize_degree(degree_text: str) -> str:
    """
    Normalize degree text to standard categories.
    Returns: 'PhD', 'Master', 'Bachelor', or 'Other'
    """
    degree_upper = degree_text.upper()
    
    if 'PHD' in degree_upper or 'PH.D' in degree_upper or 'DOCTOR' in degree_upper or 'PH D' in degree_upper:
        return 'PhD'
    elif 'MASTER' in degree_upper or 'MSC' in degree_upper or 'MA' in degree_upper or 'M.SC' in degree_upper or 'M.A' in degree_upper:
        return 'Master'
    elif 'BACHELOR' in degree_upper or 'BSC' in degree_upper or 'BA' in degree_upper or 'B.SC' in degree_upper or 'B.A' in degree_upper:
        return 'Bachelor'
    else:
        return 'Other'


def parse_max_classes(value) -> Tuple[int, str]:
    """
    Parse max_classes value from various formats.
    Returns: (parsed_value, status_message)
    """
    if pd.isna(value) or str(value).strip() == '':
        return 3, "Empty (defaulted to 3)"
    
    value_str = str(value).strip()
    
    # Handle datetime objects (dates entered by mistake)
    if isinstance(value, pd.Timestamp) or 'Timestamp' in str(type(value)):
        return 3, f"Date value '{value}' (defaulted to 3)"
    
    # Check if it looks like a date string (YYYY-MM-DD or similar with year > 1000)
    if re.search(r'\b(19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b', value_str):
        return 3, f"Date value '{value_str}' (defaulted to 3)"
    
    # Try direct integer conversion
    try:
        num = int(float(value_str))
        if num > 100:  # Likely a year or invalid number
            return 3, f"Invalid large number '{value_str}' (defaulted to 3)"
        if num > 50:  # Unreasonably high for class count
            return 3, f"Value too high '{num}' (defaulted to 3)"
        return num, "OK"
    except:
        pass
    
    # Handle ranges like "2-3" (but not dates)
    if '-' in value_str and not re.search(r'\d{4}', value_str):
        try:
            parts = value_str.split('-')
            low = int(parts[0].strip())
            high = int(parts[1].strip())
            if low <= 50 and high <= 50:  # Reasonable range for classes
                avg = (low + high) // 2
                return avg, f"Range {value_str} (using average: {avg})"
        except:
            pass
    
    # Handle "3+" or "2+"
    if '+' in value_str:
        try:
            num = int(value_str.replace('+', '').strip())
            if num <= 50:
                return num, f"'{value_str}' (using {num})"
        except:
            pass
    
    # Extract first reasonable number from the string (< 100)
    numbers = re.findall(r'\b\d+\b', value_str)
    for num_str in numbers:
        num = int(num_str)
        if 1 <= num <= 50:  # Reasonable class count (1-50)
            return num, f"Extracted {num} from '{value_str}'"
    
    # Default fallback
    return 3, f"Could not parse '{value_str}' (defaulted to 3)"


def classify_course_level(course_code: str) -> str:
    """
    Classify course as PG (Postgraduate) or UG (Undergraduate) based on course code.
    Default: All courses are PG unless specified otherwise.
    """
    # For now, all courses default to PG
    # User will have option to change in the UI
    return 'PG'


def load_file1_classes(file_path: str) -> pd.DataFrame:
    """Load and parse File 1 (Classes for T3)."""
    df = pd.read_excel(file_path, header=None)
    
    classes_data = []
    current_course = None
    
    for idx, row in df.iterrows():
        val = str(row[0]).strip()
        
        # Check if this is a course header row
        if val.startswith('ACTL') or val.startswith('RISK') or val.startswith('COMM'):
            if pd.isna(row[1]) or 'Class' not in str(row[1]):
                current_course = val
                continue
        
        # Check if this is a class row (has class number and type)
        if current_course and pd.notna(row[0]) and pd.notna(row[1]):
            class_type = str(row[1]).strip()
            
            # Only include TUT and LAB (skip LEC)
            if class_type in ['TUT', 'LAB']:
                classes_data.append({
                    'course': current_course,
                    'class_id': str(row[0]),
                    'type': class_type,
                    'section': str(row[2]) if pd.notna(row[2]) else '',
                    'mode': str(row[3]) if pd.notna(row[3]) else '',
                    'status': str(row[4]) if pd.notna(row[4]) else '',
                    'time': str(row[5]) if pd.notna(row[5]) else '',
                    'tutor': str(row[6]) if pd.notna(row[6]) else '',
                    'course_level': classify_course_level(current_course)  # Default PG
                })
    
    return pd.DataFrame(classes_data)


def load_file2_tutors(file_path: str) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, int], Dict[str, str], Dict[str, str]]:
    """
    Load and parse File 2 (Tutor Preferences).
    Returns: (tutors_df, preferences_dict, max_classes_dict, parsing_status, degrees_dict)
    """
    df = pd.read_excel(file_path, sheet_name=0)
    
    # Column indices
    first_name_col = df.columns[6]
    last_name_col = df.columns[8]
    pref_name_col = df.columns[10]
    degree_col = df.columns[57]  # BF column (index 57, 0-based)
    t3_pref_col = df.columns[108]  # DE - T3 preferences
    max_classes_col = df.columns[110]  # DG - Max classes (tutorials per session)
    
    tutors_data = []
    preferences = {}
    max_classes = {}
    parsing_status = {}
    degrees = {}
    
    for idx, row in df.iterrows():
        # Get tutor name
        pref_name = row[pref_name_col]
        if pd.isna(pref_name) or str(pref_name).strip() == '':
            first = str(row[first_name_col]).strip() if pd.notna(row[first_name_col]) else ''
            last = str(row[last_name_col]).strip() if pd.notna(row[last_name_col]) else ''
            tutor_name = f"{first} {last}".strip()
        else:
            tutor_name = str(pref_name).strip()
        
        if not tutor_name or tutor_name == '':
            continue
        
        # Get degree information from column BF
        degree_info = extract_degree(row[degree_col])
        
        # Get T3 preferences
        t3_pref_text = row[t3_pref_col]
        t3_courses_raw = extract_course_codes(t3_pref_text)
        
        # Expand dual codes (e.g., "ACTL3162/5106" -> ["ACTL3162", "ACTL5106"])
        t3_courses_expanded = []
        for code in t3_courses_raw:
            if '/' in code:
                # Split dual code: "ACTL3162/5106" -> ["ACTL3162", "ACTL5106"]
                parts = code.split('/')
                base = parts[0]  # e.g., "ACTL3162"
                prefix = base[:4]  # e.g., "ACTL"
                second_code = prefix + parts[1]  # e.g., "ACTL5106"
                t3_courses_expanded.append(base)
                t3_courses_expanded.append(second_code)
            else:
                t3_courses_expanded.append(code)
        
        # Remove duplicates
        t3_courses_expanded = list(set(t3_courses_expanded))
        
        # Only include tutors who have T3 preferences
        if len(t3_courses_expanded) > 0:
            # Parse max classes
            max_val, status = parse_max_classes(row[max_classes_col])
            
            tutors_data.append({
                'tutor_name': tutor_name,
                'first_name': str(row[first_name_col]) if pd.notna(row[first_name_col]) else '',
                'last_name': str(row[last_name_col]) if pd.notna(row[last_name_col]) else '',
                'email': str(row[df.columns[9]]) if pd.notna(row[df.columns[9]]) else '',
                'degree': degree_info,
                't3_preferences_raw': str(t3_pref_text),
                't3_courses': ', '.join(sorted(t3_courses_expanded)),
                'max_classes': max_val,
                'max_classes_status': status
            })
            
            preferences[tutor_name] = t3_courses_expanded
            max_classes[tutor_name] = max_val
            parsing_status[tutor_name] = status
            degrees[tutor_name] = degree_info
    
    return pd.DataFrame(tutors_data), preferences, max_classes, parsing_status, degrees


def main():
    st.set_page_config(
        page_title="Tutor Assignment Optimizer",
        page_icon="👨‍🏫",
        layout="wide"
    )
    
    st.title("👨‍🏫 Tutor-Class Assignment Optimizer (T3)")
    st.markdown("**Linear Programming Solution with Time Conflict Detection & Degree Requirements**")
    st.markdown("---")
    
    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
    
    # Step 1: File Upload
    if st.session_state.step == 1:
        show_upload_step()
    
    # Step 2: Set Course Levels (PG/UG)
    elif st.session_state.step == 2:
        show_course_level_step()
    
    # Step 3: Review Tutors (Degrees & Max Classes)
    elif st.session_state.step == 3:
        show_review_tutors_step()
    
    # Step 4: Analysis & Visualization
    elif st.session_state.step == 4:
        show_analysis_step()
    
    # Step 5: Run Optimization & Results
    elif st.session_state.step == 5:
        show_optimization_step()


def show_upload_step():
    """Step 1: Upload both files."""
    st.header("Step 1: Upload Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 File 1: Classes (T3)")
        st.markdown("""
        **Required structure:**
        - Course codes in first column
        - Class details: Number, Type (TUT/LAB), Section, Mode, Status, Time, Tutor
        - Only TUT/LAB classes will be used (LEC ignored)
        - **All courses default to PG (Postgraduate)**
        """)
        
        file1 = st.file_uploader(
            "Upload File 1 (Classes)",
            type=['xlsx', 'xls'],
            key='file1'
        )
    
    with col2:
        st.subheader("👥 File 2: Tutor Preferences")
        st.markdown("""
        **Will extract:**
        - Column BF (57): Degree (PhD/Master/Bachelor)
        - Column DE (108): T3 course preferences
        - Column DG (110): Maximum classes per week
        - Tutor names (First, Last, Preferred)
        
        **Degree Rules:**
        - 🎓 PhD → Can teach both PG and UG courses
        - 📚 Non-PhD → Can only teach UG courses
        """)
        
        file2 = st.file_uploader(
            "Upload File 2 (Tutor Preferences)",
            type=['xlsx', 'xls'],
            key='file2'
        )
    
    if file1 is not None and file2 is not None:
        with st.spinner("Processing files..."):
            try:
                # Load classes
                classes_df = load_file1_classes(file1)
                st.session_state.classes_df = classes_df
                
                # Load tutors
                tutors_df, preferences, max_classes, parsing_status, degrees = load_file2_tutors(file2)
                st.session_state.tutors_df = tutors_df
                st.session_state.preferences = preferences
                st.session_state.max_classes = max_classes
                st.session_state.parsing_status = parsing_status
                st.session_state.degrees = degrees
                
                # Show preview
                st.success("✅ Files loaded successfully!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Classes Found (TUT/LAB)", len(classes_df))
                with col2:
                    st.metric("Unique Courses", classes_df['course'].nunique())
                with col3:
                    st.metric("Tutors with T3 Preferences", len(tutors_df))
                
                # Degree distribution
                st.subheader("🎓 Tutor Degree Distribution")
                degree_counts = {}
                for degree in degrees.values():
                    normalized = normalize_degree(degree)
                    degree_counts[normalized] = degree_counts.get(normalized, 0) + 1
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("PhD Tutors", degree_counts.get('PhD', 0))
                    st.caption("Can teach PG & UG")
                with col2:
                    st.metric("Master Tutors", degree_counts.get('Master', 0))
                    st.caption("Can teach UG only")
                with col3:
                    st.metric("Bachelor Tutors", degree_counts.get('Bachelor', 0))
                    st.caption("Can teach UG only")
                with col4:
                    st.metric("Other", degree_counts.get('Other', 0))
                    st.caption("Can teach UG only")
                
                # Show preview
                with st.expander("📋 Preview Classes Data"):
                    st.dataframe(classes_df.head(20), use_container_width=True)
                
                with st.expander("👥 Preview Tutors Data"):
                    st.dataframe(tutors_df[['tutor_name', 'degree', 't3_courses', 'max_classes', 'max_classes_status']].head(20), 
                               use_container_width=True)
                
                if st.button("Next: Set Course Levels (PG/UG) →", type="primary"):
                    st.session_state.step = 2
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error processing files: {str(e)}")
                st.exception(e)


def show_course_level_step():
    """Step 2: Set course levels (PG/UG)."""
    st.header("Step 2: Set Course Levels (PG/UG)")
    
    classes_df = st.session_state.classes_df
    
    st.markdown("""
    **All courses are set to PG (Postgraduate) by default.**  
    Change any course to UG (Undergraduate) if needed.
    
    **Important:**
    - 🎓 PhD tutors can teach **both PG and UG** courses
    - 📚 Non-PhD tutors can **only teach UG** courses
    """)
    
    st.markdown("---")
    
    # Get unique courses
    unique_courses = sorted(classes_df['course'].unique())
    
    st.subheader("Set Course Levels")
    
    # Create a dictionary to store course levels
    if 'course_levels' not in st.session_state:
        st.session_state.course_levels = {course: 'PG' for course in unique_courses}
    
    # Display in columns for better layout
    col1, col2 = st.columns(2)
    
    for idx, course in enumerate(unique_courses):
        with col1 if idx % 2 == 0 else col2:
            current_level = st.session_state.course_levels.get(course, 'PG')
            new_level = st.radio(
                f"**{course}**",
                options=['PG', 'UG'],
                index=0 if current_level == 'PG' else 1,
                key=f"level_{course}",
                horizontal=True
            )
            st.session_state.course_levels[course] = new_level
            
            # Show class count
            num_classes = len(classes_df[classes_df['course'] == course])
            st.caption(f"Classes: {num_classes}")
            st.markdown("---")
    
    # Summary
    st.subheader("📊 Summary")
    pg_courses = [c for c, l in st.session_state.course_levels.items() if l == 'PG']
    ug_courses = [c for c, l in st.session_state.course_levels.items() if l == 'UG']
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("PG Courses", len(pg_courses))
        if pg_courses:
            st.caption(", ".join(pg_courses))
    with col2:
        st.metric("UG Courses", len(ug_courses))
        if ug_courses:
            st.caption(", ".join(ug_courses))
    
    st.markdown("---")
    
    # Navigation
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("← Back to Upload"):
            st.session_state.step = 1
            st.rerun()
    
    with col2:
        if st.button("Next: Review Tutors →", type="primary"):
            # Update classes_df with course levels
            classes_df['course_level'] = classes_df['course'].map(st.session_state.course_levels)
            st.session_state.classes_df = classes_df
            st.session_state.step = 3
            st.rerun()


def show_review_tutors_step():
    """Step 3: Review and edit tutor information (degrees and max classes)."""
    st.header("Step 3: Review & Edit Tutor Information")
    
    tutors_df = st.session_state.tutors_df
    parsing_status = st.session_state.parsing_status
    degrees = st.session_state.degrees
    
    st.markdown("""
    **Review and edit tutor information below:**
    - 🎓 **Education Level**: PhD tutors can teach PG & UG; Non-PhD can only teach UG
    - 📊 **Max Classes**: Maximum number of classes per tutor
    
    Edit any values as needed.
    """)
    
    st.markdown("---")
    
    # Check for issues
    needs_review_max = tutors_df[tutors_df['max_classes_status'].str.contains('defaulted|Could not parse', case=False, na=False)]
    
    if len(needs_review_max) > 0:
        st.warning(f"⚠️ {len(needs_review_max)} tutor(s) have max_classes values that may need correction")
    else:
        st.success("✅ All max_classes values parsed successfully!")
    
    st.markdown("---")
    
    # Editable table
    st.subheader("Edit Tutor Information")
    
    edited_degrees = {}
    edited_max_classes = {}
    
    # Create scrollable container
    for idx, row in tutors_df.iterrows():
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 3, 1])
            
            with col1:
                st.markdown(f"**{row['tutor_name']}**")
                st.caption(f"Preferences: {row['t3_courses'][:50]}...")
            
            with col2:
                # Degree dropdown
                current_degree = degrees.get(row['tutor_name'], 'Other')
                normalized_degree = normalize_degree(current_degree)
                
                degree_options = ['PhD', 'Master', 'Bachelor', 'Other']
                default_idx = degree_options.index(normalized_degree) if normalized_degree in degree_options else 3
                
                new_degree = st.selectbox(
                    "Education",
                    options=degree_options,
                    index=default_idx,
                    key=f"degree_{idx}",
                    label_visibility="collapsed"
                )
                
                # Show teaching eligibility
                if new_degree == 'PhD':
                    st.caption("✅ Can teach PG & UG")
                else:
                    st.caption("⚠️ Can teach UG only")
                
                edited_degrees[row['tutor_name']] = new_degree
            
            with col3:
                # Show original degree text
                original_text = row['degree'][:30] + "..." if len(row['degree']) > 30 else row['degree']
                st.caption(f"Original: {original_text}")
            
            with col4:
                # Max classes input
                status = row['max_classes_status']
                
                # Status indicator
                if 'defaulted' in status.lower() or 'could not parse' in status.lower():
                    st.caption("❌ " + status[:40])
                elif 'range' in status.lower() or 'extracted' in status.lower():
                    st.caption("⚠️ " + status[:40])
                else:
                    st.caption("✅ " + status[:40])
            
            with col5:
                new_max = st.number_input(
                    "Max",
                    min_value=1,
                    max_value=50,
                    value=int(row['max_classes']),
                    key=f"max_{idx}",
                    label_visibility="collapsed"
                )
                edited_max_classes[row['tutor_name']] = new_max
            
            st.markdown("---")
    
    # Summary of changes
    st.subheader("📊 Summary of Education Levels")
    
    degree_summary = {}
    for degree in edited_degrees.values():
        degree_summary[degree] = degree_summary.get(degree, 0) + 1
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("PhD", degree_summary.get('PhD', 0))
        st.caption("Can teach PG & UG")
    with col2:
        st.metric("Master", degree_summary.get('Master', 0))
        st.caption("Can teach UG only")
    with col3:
        st.metric("Bachelor", degree_summary.get('Bachelor', 0))
        st.caption("Can teach UG only")
    with col4:
        st.metric("Other", degree_summary.get('Other', 0))
        st.caption("Can teach UG only")
    
    st.markdown("---")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Back to Course Levels"):
            st.session_state.step = 2
            st.rerun()
    
    with col3:
        if st.button("Next: Analysis →", type="primary"):
            # Update session state with edited values
            st.session_state.degrees = edited_degrees
            st.session_state.max_classes = edited_max_classes
            st.session_state.step = 4
            st.rerun()


def show_analysis_step():
    """Step 4: Show data analysis and visualizations."""
    st.header("Step 4: Data Analysis")
    
    classes_df = st.session_state.classes_df
    tutors_df = st.session_state.tutors_df
    preferences = st.session_state.preferences
    max_classes = st.session_state.max_classes
    degrees = st.session_state.degrees
    
    # Summary metrics
    st.subheader("📊 Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Classes", len(classes_df))
    with col2:
        st.metric("Unique Courses", classes_df['course'].nunique())
    with col3:
        st.metric("Available Tutors", len(tutors_df))
    with col4:
        total_capacity = sum(max_classes.values())
        st.metric("Total Tutor Capacity", total_capacity)
    with col5:
        phd_count = sum(1 for d in degrees.values() if d == 'PhD')
        st.metric("PhD Tutors", phd_count)
    
    # Check feasibility
    if total_capacity < len(classes_df):
        st.warning(f"⚠️ Total tutor capacity ({total_capacity}) is less than total classes ({len(classes_df)}). Optimization will proceed and show unassigned classes.")
    else:
        surplus = total_capacity - len(classes_df)
        st.success(f"✅ Sufficient capacity: {surplus} extra class slots available")
    
    st.markdown("---")
    
    # Course level distribution
    st.subheader("📚 Course Level Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        level_counts = classes_df.groupby('course_level').agg({
            'course': 'nunique',
            'class_id': 'count'
        }).reset_index()
        level_counts.columns = ['Level', 'Unique Courses', 'Total Classes']
        
        st.dataframe(level_counts, use_container_width=True, hide_index=True)
    
    with col2:
        fig_level = px.pie(
            classes_df,
            names='course_level',
            title='Classes by Level',
            color='course_level',
            color_discrete_map={'PG': '#FF6B6B', 'UG': '#4ECDC4'}
        )
        st.plotly_chart(fig_level, use_container_width=True)
    
    # Classes per course
    st.subheader("📊 Classes per Course")
    course_details = classes_df.groupby(['course', 'course_level']).size().reset_index(name='count')
    course_details = course_details.sort_values('count', ascending=False)
    
    fig_courses = px.bar(
        course_details,
        x='course',
        y='count',
        color='course_level',
        title='Number of Classes (TUT/LAB) per Course',
        labels={'course': 'Course', 'count': 'Number of Classes', 'course_level': 'Level'},
        color_discrete_map={'PG': '#FF6B6B', 'UG': '#4ECDC4'}
    )
    fig_courses.update_layout(height=400)
    st.plotly_chart(fig_courses, use_container_width=True)
    
    # Course coverage analysis with degree constraints
    st.subheader("📋 Course Coverage Analysis (with Degree Requirements)")
    
    st.info("ℹ️ Even if some courses show insufficient capacity, optimization will still run and show which classes remain unassigned.")
    
    coverage_data = []
    courses = sorted(classes_df['course'].unique())
    
    for course in courses:
        course_df = classes_df[classes_df['course'] == course]
        num_classes = len(course_df)
        course_level = course_df.iloc[0]['course_level']
        
        # Count qualified tutors based on degree requirements
        if course_level == 'PG':
            # Only PhD tutors can teach PG
            qualified_tutors = [
                tutor for tutor in tutors_df['tutor_name'].unique()
                if course in preferences.get(tutor, []) and degrees.get(tutor, '') == 'PhD'
            ]
        else:  # UG
            # All tutors can teach UG
            qualified_tutors = [
                tutor for tutor in tutors_df['tutor_name'].unique()
                if course in preferences.get(tutor, [])
            ]
        
        num_qualified = len(qualified_tutors)
        total_capacity_for_course = sum([max_classes.get(tutor, 0) for tutor in qualified_tutors])
        
        coverage_data.append({
            'Course': course,
            'Level': course_level,
            'Classes': num_classes,
            'Qualified Tutors': num_qualified,
            'Tutor Capacity': total_capacity_for_course,
            'Ratio': f"{total_capacity_for_course / num_classes:.2f}x" if num_classes > 0 else "N/A",
            'Status': '✅' if total_capacity_for_course >= num_classes else '⚠️'
        })
    
    coverage_df = pd.DataFrame(coverage_data)
    st.dataframe(coverage_df, use_container_width=True, hide_index=True)
    
    # Highlight problematic courses
    problematic = coverage_df[coverage_df['Status'] == '⚠️']
    if len(problematic) > 0:
        st.warning(f"⚠️ {len(problematic)} course(s) have insufficient qualified tutor capacity:")
        st.dataframe(problematic, use_container_width=True, hide_index=True)
        
        # Provide specific reasons
        for idx, row in problematic.iterrows():
            if row['Level'] == 'PG' and row['Qualified Tutors'] == 0:
                st.error(f"**{row['Course']}**: No PhD tutors available who prefer this course!")
            elif row['Qualified Tutors'] == 0:
                st.error(f"**{row['Course']}**: No tutors have expressed preference for this course!")
        
        st.info("💡 The optimization will still run and show which specific classes cannot be assigned.")
    
    st.markdown("---")
    
    # Navigation
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to Review Tutors"):
            st.session_state.step = 3
            st.rerun()
    
    with col2:
        if st.button("Run Optimization →", type="primary"):
            st.session_state.step = 5
            st.rerun()


def show_optimization_step():
    """Step 5: Run optimization and show results."""
    st.header("Step 5: Optimization Results")
    
    classes_df = st.session_state.classes_df
    tutors_df = st.session_state.tutors_df
    preferences = st.session_state.preferences
    max_classes = st.session_state.max_classes
    degrees = st.session_state.degrees
    
    # Optimization parameters
    st.subheader("⚙️ Optimization Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        course_diversity_penalty = st.slider(
            "Course Diversity Penalty",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            help="Higher penalty encourages tutors to teach fewer different courses (1-2 courses preferred)"
        )
    
    with col2:
        st.info(f"""
        **Current Setting: {course_diversity_penalty}**
        
        - **0**: No penalty (tutors can teach many courses)
        - **5**: Moderate penalty (recommended)
        - **10+**: Strong penalty (tutors prefer 1-2 courses only)
        """)
    
    st.markdown("---")
    
    # Build preference dictionary for LP
    pref_dict = {}
    for tutor, courses in preferences.items():
        for course in courses:
            pref_dict[(tutor, course)] = 10  # All preferences = 10
    
    with st.spinner("🔄 Running Linear Programming optimization with degree constraints and course diversity penalty..."):
        try:
            # Create and solve LP
            lp = TutorAssignmentLP(
                classes_df=classes_df,
                tutors_df=tutors_df,
                preferences=pref_dict,
                tutor_max_classes=max_classes,
                tutor_degrees=degrees,
                course_diversity_penalty=course_diversity_penalty
            )
            
            solution = lp.solve()
            
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Status", solution['status'])
            with col2:
                if solution['objective_value']:
                    st.metric("Objective Value", f"{solution['objective_value']:.0f}")
            with col3:
                assigned_count = len(classes_df) - len(solution['unassigned_classes'])
                st.metric("Assigned Classes", assigned_count)
            with col4:
                unassigned_count = len(solution['unassigned_classes'])
                st.metric("Unassigned Classes", unassigned_count)
            
            if solution['status'] == 'Optimal':
                if unassigned_count == 0:
                    st.success("✅ Optimization completed successfully! All classes assigned!")
                else:
                    st.warning(f"⚠️ Optimization completed. {unassigned_count} classes could not be assigned (see details below).")
                
                # Create results dataframe
                results_data = []
                for idx, row in classes_df.iterrows():
                    course = row['course']
                    class_id = row['class_id']
                    assigned_tutor = solution['assignments'].get((course, class_id), 'UNASSIGNED')
                    
                    # Get tutor degree if assigned
                    tutor_degree = degrees.get(assigned_tutor, 'N/A') if assigned_tutor != 'UNASSIGNED' else 'N/A'
                    
                    results_data.append({
                        'Course': course,
                        'Level': row['course_level'],
                        'Class ID': class_id,
                        'Type': row['type'],
                        'Section': row['section'],
                        'Time': row['time'],
                        'Assigned Tutor': assigned_tutor,
                        'Tutor Degree': tutor_degree
                    })
                
                results_df = pd.DataFrame(results_data)
                
                # Show assignment table
                st.subheader("📋 Class Assignments")
                
                # Filter options
                col1, col2, col3 = st.columns(3)
                with col1:
                    filter_course = st.selectbox(
                        "Filter by Course:",
                        options=['All'] + sorted(classes_df['course'].unique().tolist())
                    )
                with col2:
                    filter_tutor = st.selectbox(
                        "Filter by Tutor:",
                        options=['All', 'UNASSIGNED'] + sorted([t for t in tutors_df['tutor_name'].unique()])
                    )
                with col3:
                    filter_level = st.selectbox(
                        "Filter by Level:",
                        options=['All', 'PG', 'UG']
                    )
                
                display_df = results_df.copy()
                if filter_course != 'All':
                    display_df = display_df[display_df['Course'] == filter_course]
                if filter_tutor != 'All':
                    display_df = display_df[display_df['Assigned Tutor'] == filter_tutor]
                if filter_level != 'All':
                    display_df = display_df[display_df['Level'] == filter_level]
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Tutor workload summary
                st.subheader("👥 Tutor Workload Summary")
                
                workload_data = []
                for tutor in tutors_df['tutor_name'].unique():
                    load = solution['tutor_loads'][tutor]
                    assigned_classes = load['classes']
                    
                    # Group by course and level
                    courses_count = {}
                    pg_count = 0
                    ug_count = 0
                    
                    for course, class_id in assigned_classes:
                        courses_count[course] = courses_count.get(course, 0) + 1
                        # Get level
                        class_level = classes_df[
                            (classes_df['course'] == course) & 
                            (classes_df['class_id'] == class_id)
                        ].iloc[0]['course_level']
                        if class_level == 'PG':
                            pg_count += 1
                        else:
                            ug_count += 1
                    
                    courses_str = ', '.join([f"{course}({count})" for course, count in courses_count.items()])
                    num_different_courses = solution['tutor_course_diversity'].get(tutor, 0)
                    
                    tutor_degree = degrees.get(tutor, 'Not Specified')
                    
                    workload_data.append({
                        'Tutor': tutor,
                        'Degree': tutor_degree,
                        'Different Courses': num_different_courses,
                        'PG Classes': pg_count,
                        'UG Classes': ug_count,
                        'Total Classes': load['total'],
                        'Max Allowed': max_classes.get(tutor, 0),
                        'Utilization': f"{load['total']}/{max_classes.get(tutor, 0)}",
                        'Courses Assigned': courses_str if courses_str else 'None'
                    })
                
                workload_df = pd.DataFrame(workload_data)
                workload_df = workload_df.sort_values('Total Classes', ascending=False)
                st.dataframe(workload_df, use_container_width=True, hide_index=True)
                
                # Course diversity analysis
                st.subheader("📊 Course Diversity Analysis")
                diversity_summary = workload_df[workload_df['Total Classes'] > 0]['Different Courses'].value_counts().sort_index()
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write("**Tutors by Number of Different Courses:**")
                    for num_courses, count in diversity_summary.items():
                        st.write(f"- Teaching {num_courses} course(s): {count} tutor(s)")
                
                with col2:
                    fig_diversity = px.bar(
                        x=diversity_summary.index,
                        y=diversity_summary.values,
                        labels={'x': 'Number of Different Courses', 'y': 'Number of Tutors'},
                        title='Distribution of Course Diversity',
                        color=diversity_summary.values,
                        color_continuous_scale='Blues'
                    )
                    fig_diversity.update_layout(showlegend=False)
                    st.plotly_chart(fig_diversity, use_container_width=True)
                
                # Visualization
                st.subheader("📈 Workload Visualization")
                fig_workload = px.bar(
                    workload_df[workload_df['Total Classes'] > 0],
                    x='Tutor',
                    y=['PG Classes', 'UG Classes'],
                    title='Tutor Workload Distribution by Course Level',
                    labels={'value': 'Number of Classes', 'variable': 'Level'},
                    color_discrete_map={'PG Classes': '#FF6B6B', 'UG Classes': '#4ECDC4'}
                )
                fig_workload.update_layout(height=400, xaxis_tickangle=45)
                st.plotly_chart(fig_workload, use_container_width=True)
                
                # Unassigned classes
                if solution['unassigned_classes']:
                    st.subheader("⚠️ Unassigned Classes")
                    st.error(f"The following {len(solution['unassigned_classes'])} classes could not be assigned:")
                    
                    unassigned_data = []
                    for course, class_id in solution['unassigned_classes']:
                        class_row = classes_df[
                            (classes_df['course'] == course) & 
                            (classes_df['class_id'] == class_id)
                        ].iloc[0]
                        
                        course_level = class_row['course_level']
                        
                        # Find why it couldn't be assigned
                        if course_level == 'PG':
                            qualified_tutors = [
                                t for t, courses_pref in preferences.items() 
                                if course in courses_pref and degrees.get(t, '') == 'PhD'
                            ]
                        else:
                            qualified_tutors = [t for t, courses_pref in preferences.items() if course in courses_pref]
                        
                        if len(qualified_tutors) == 0:
                            if course_level == 'PG':
                                reason = "No PhD tutors with preference for this course"
                            else:
                                reason = "No tutors with preference for this course"
                        else:
                            reason = "Time conflict or capacity exceeded"
                        
                        unassigned_data.append({
                            'Course': course,
                            'Level': course_level,
                            'Class ID': class_id,
                            'Section': class_row['section'],
                            'Time': class_row['time'],
                            'Qualified Tutors': len(qualified_tutors),
                            'Possible Reason': reason
                        })
                    
                    unassigned_df = pd.DataFrame(unassigned_data)
                    st.dataframe(unassigned_df, use_container_width=True, hide_index=True)
                    
                    # Group unassigned by course
                    st.subheader("📋 Unassigned Classes by Course")
                    unassigned_by_course = unassigned_df.groupby(['Course', 'Level']).size().reset_index(name='Unassigned Count')
                    st.dataframe(unassigned_by_course, use_container_width=True, hide_index=True)
                else:
                    st.success("🎉 All classes successfully assigned!")
                
                # Download results
                st.subheader("📥 Download Results")
                
                # Create Excel output
                from io import BytesIO
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    results_df.to_excel(writer, sheet_name='Assignments', index=False)
                    workload_df.to_excel(writer, sheet_name='Tutor Workload', index=False)
                    if solution['unassigned_classes']:
                        unassigned_df.to_excel(writer, sheet_name='Unassigned', index=False)
                        unassigned_by_course.to_excel(writer, sheet_name='Unassigned by Course', index=False)
                
                st.download_button(
                    label="📥 Download Results (Excel)",
                    data=output.getvalue(),
                    file_name="tutor_assignments_T3_optimized.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
            else:
                st.error(f"❌ Optimization failed: {solution['status']}")
                st.write("Possible reasons:")
                st.write("- Model is infeasible due to constraints")
                st.write("- Try adjusting course diversity penalty")
                st.write("- Check degree requirements and tutor availability")
            
        except Exception as e:
            st.error(f"❌ Error during optimization: {str(e)}")
            st.exception(e)
    
    # Navigation
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to Analysis"):
            st.session_state.step = 4
            st.rerun()
    
    with col2:
        if st.button("Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()
