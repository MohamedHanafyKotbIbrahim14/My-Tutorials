import streamlit as st
import pandas as pd
import numpy as np
import pulp
import re
from typing import Dict, List, Tuple, Set
import plotly.express as px
import plotly.graph_objects as go
from datetime import time as dt_time
import time as time_module

class TutorAssignmentLP:
    """Tutor-Class Assignment using Linear Programming with dual preferences, time conflict detection and degree requirements."""
    
    def __init__(self, classes_df: pd.DataFrame, tutors_df: pd.DataFrame, 
                 tutor_preferences: Dict[Tuple[str, str], float],
                 academic_preferences: Dict[Tuple[str, str], float],
                 tutor_max_classes: Dict[str, int],
                 tutor_degrees: Dict[str, str],
                 w1: float = 10.0,
                 w2: float = 20.0,
                 theta: float = 2.0,
                 course_diversity_penalty: float = 5.0,
                 phd_priority_bonus: float = 10.0,
                 master_priority_bonus: float = 5.0):
        """
        Initialize the LP problem with dual preferences.
        
        Parameters:
        - tutor_preferences: Dict[(tutor, course)] = binary (0 or 10)
        - academic_preferences: Dict[(tutor, course)] = rating (1-10, default 5)
        - w1: Weight for tutor preference (default: 10.0)
        - w2: Weight for academic preference (default: 20.0)
        - theta: Academic veto threshold (default: 2.0)
        """
        self.classes_df = classes_df
        self.tutors_df = tutors_df
        self.tutor_preferences = tutor_preferences
        self.academic_preferences = academic_preferences
        self.tutor_max_classes = tutor_max_classes
        self.tutor_degrees = tutor_degrees
        self.w1 = w1
        self.w2 = w2
        self.theta = theta
        self.course_diversity_penalty = course_diversity_penalty
        self.phd_priority_bonus = phd_priority_bonus
        self.master_priority_bonus = master_priority_bonus
        
        self.tutors = list(tutors_df['tutor_name'].unique())
        self.courses = list(classes_df['course'].unique())
        
        self.model = None
        self.x_vars = {}
        self.y_vars = {}
        self.time_conflicts = self._detect_time_conflicts()
        
        # Pre-calculate eligible assignments
        self.eligible_assignments = self._get_eligible_assignments()
        
    def _get_eligible_assignments(self) -> List[Tuple[str, str, str]]:
        """Pre-calculate all eligible (tutor, course, class_id) combinations using E_tc rule."""
        eligible = []
        
        for tutor in self.tutors:
            for idx, row in self.classes_df.iterrows():
                course = row['course']
                class_id = row['class_id']
                
                # Check eligibility: E_tc = 1 if p_tc = 10 AND a_tc > theta AND degree-eligible
                p_tc = self.tutor_preferences.get((tutor, course), 0)
                a_tc = self.academic_preferences.get((tutor, course), 5.0)
                
                # Tutor must have expressed preference (p_tc = 10)
                if p_tc <= 0:
                    continue
                
                # Academic veto: a_tc must be > theta
                if a_tc <= self.theta:
                    continue
                
                # Check degree eligibility
                if not self._can_tutor_teach_course(tutor, course, row):
                    continue
                
                eligible.append((tutor, course, class_id))
        
        return eligible
        
    def _detect_time_conflicts(self) -> Set[Tuple[str, str]]:
        """Detect time conflicts between classes - returns set of conflict pairs."""
        conflicts = set()
        
        classes_with_times = []
        for idx, row in self.classes_df.iterrows():
            time_str = str(row['time'])
            parsed_times = self._parse_time_string(time_str)
            if parsed_times:
                classes_with_times.append({
                    'class_id': row['class_id'],
                    'times': parsed_times
                })
        
        for i, class1 in enumerate(classes_with_times):
            for class2 in classes_with_times[i+1:]:
                if self._times_overlap(class1['times'], class2['times']):
                    conflicts.add((class1['class_id'], class2['class_id']))
                    conflicts.add((class2['class_id'], class1['class_id']))
        
        return conflicts
    
    def _parse_time_string(self, time_str: str) -> List[Dict]:
        """Parse time string like 'Fri 09-10:30'."""
        if pd.isna(time_str) or time_str == 'nan':
            return []
        
        time_slots = []
        pattern = r'(Mon|Tue|Wed|Thu|Fri)\s+(\d{1,2})(?::(\d{2}))?-(\d{1,2})(?::(\d{2}))?'
        
        for match in re.finditer(pattern, time_str):
            day = match.group(1)
            start_hour = int(match.group(2))
            start_min = int(match.group(3)) if match.group(3) else 0
            end_hour = int(match.group(4))
            end_min = int(match.group(5)) if match.group(5) else 0
            
            time_slots.append({
                'day': day,
                'start': start_hour * 60 + start_min,
                'end': end_hour * 60 + end_min
            })
        
        return time_slots
    
    def _times_overlap(self, times1: List[Dict], times2: List[Dict]) -> bool:
        """Check if two time slot lists overlap."""
        for t1 in times1:
            for t2 in times2:
                if t1['day'] == t2['day']:
                    if not (t1['end'] <= t2['start'] or t2['end'] <= t1['start']):
                        return True
        return False
    
    def _can_tutor_teach_course(self, tutor: str, course: str, class_row: pd.Series) -> bool:
        """Check if tutor can teach based on degree."""
        tutor_degree = self.tutor_degrees.get(tutor, "")
        course_level = class_row['course_level']
        
        if tutor_degree == 'PhD':
            return True
        
        if course_level == 'UG':
            return True
        
        return False
    
    def build_model(self):
        """Build the LP model with dual preferences - optimized version."""
        self.model = pulp.LpProblem("Tutor_Assignment_Dual_Preferences", pulp.LpMaximize)
        
        # Create decision variables only for eligible assignments
        for tutor, course, class_id in self.eligible_assignments:
            var_name = f"x_{len(self.x_vars)}"  # Shorter variable names
            self.x_vars[(tutor, course, class_id)] = pulp.LpVariable(var_name, cat='Binary')
        
        # Create course indicator variables
        tutor_course_pairs = set((t, c) for t, c, _ in self.eligible_assignments)
        for tutor, course in tutor_course_pairs:
            var_name = f"y_{len(self.y_vars)}"
            self.y_vars[(tutor, course)] = pulp.LpVariable(var_name, cat='Binary')
        
        # Objective function: Combined preferences + degree bonuses - diversity penalty
        
        # Combined preference term: (w1 * p_tc + w2 * a_tc) * x_tck
        combined_preference_term = pulp.lpSum([
            (self.w1 * self.tutor_preferences.get((tutor, course), 0) + 
             self.w2 * self.academic_preferences.get((tutor, course), 5.0)) * 
            self.x_vars[(tutor, course, class_id)]
            for (tutor, course, class_id) in self.x_vars.keys()
        ])
        
        # PhD bonus (highest priority)
        phd_bonus_term = self.phd_priority_bonus * pulp.lpSum([
            self.x_vars[(tutor, course, class_id)]
            for (tutor, course, class_id) in self.x_vars.keys()
            if self.tutor_degrees.get(tutor, '') == 'PhD'
        ])
        
        # Master bonus (medium priority)
        master_bonus_term = self.master_priority_bonus * pulp.lpSum([
            self.x_vars[(tutor, course, class_id)]
            for (tutor, course, class_id) in self.x_vars.keys()
            if self.tutor_degrees.get(tutor, '') == 'Master'
        ])
        
        # Course diversity penalty
        diversity_penalty_term = self.course_diversity_penalty * pulp.lpSum([
            self.y_vars[(tutor, course)]
            for (tutor, course) in self.y_vars.keys()
        ])
        
        self.model += combined_preference_term + phd_bonus_term + master_bonus_term - diversity_penalty_term
        
        self._add_constraints()
    
    def _add_constraints(self):
        """Add constraints - optimized version."""
        
        # 1. Class coverage (at most 1 tutor per class)
        class_assignments = {}
        for tutor, course, class_id in self.x_vars.keys():
            key = (course, class_id)
            if key not in class_assignments:
                class_assignments[key] = []
            class_assignments[key].append(self.x_vars[(tutor, course, class_id)])
        
        for (course, class_id), vars_list in class_assignments.items():
            self.model += (pulp.lpSum(vars_list) <= 1, f"C_{course}_{class_id}")
        
        # 2. Tutor workload limits
        tutor_assignments = {}
        for tutor, course, class_id in self.x_vars.keys():
            if tutor not in tutor_assignments:
                tutor_assignments[tutor] = []
            tutor_assignments[tutor].append(self.x_vars[(tutor, course, class_id)])
        
        for tutor, vars_list in tutor_assignments.items():
            max_load = self.tutor_max_classes.get(tutor, 3)
            self.model += (pulp.lpSum(vars_list) <= max_load, f"W_{tutor}")
        
        # 3. Time conflicts
        for tutor in self.tutors:
            tutor_classes = [(c, cl) for (t, c, cl) in self.x_vars.keys() if t == tutor]
            
            for i, (course1, class1) in enumerate(tutor_classes):
                for course2, class2 in tutor_classes[i+1:]:
                    if (class1, class2) in self.time_conflicts:
                        self.model += (
                            self.x_vars[(tutor, course1, class1)] + 
                            self.x_vars[(tutor, course2, class2)] <= 1,
                            f"T_{tutor}_{len(self.model.constraints)}"
                        )
        
        # 4. Link y to x variables
        for tutor, course in self.y_vars.keys():
            relevant_x = [
                self.x_vars[(t, c, cl)]
                for (t, c, cl) in self.x_vars.keys()
                if t == tutor and c == course
            ]
            
            if relevant_x:
                # y must be at least as large as any x
                for x_var in relevant_x:
                    self.model += (self.y_vars[(tutor, course)] >= x_var)
    
    def solve(self, time_limit=None):
        """Solve the optimization problem."""
        if self.model is None:
            self.build_model()
        
        # Configure solver - no time limit if None
        solver_params = {
            'msg': 0,  # Suppress output
            'gapRel': 0.02,  # Stop if within 2% of optimal
            'threads': 4  # Use multiple threads
        }
        
        # Only add time limit if specified
        if time_limit is not None:
            solver_params['timeLimit'] = time_limit
        
        solver = pulp.PULP_CBC_CMD(**solver_params)
        
        start_time = time_module.time()
        self.model.solve(solver)
        solve_time = time_module.time() - start_time
        
        status = pulp.LpStatus[self.model.status]
        
        if status in ['Optimal', 'Not Solved']:
            solution = self._extract_solution()
            solution['solve_time'] = solve_time
            return solution
        else:
            return {
                'status': status,
                'objective_value': None,
                'assignments': {},
                'tutor_loads': {},
                'unassigned_classes': [],
                'tutor_course_diversity': {},
                'solve_time': solve_time
            }
    
    def _extract_solution(self):
        """Extract solution from solved model."""
        assignments = {}
        tutor_loads = {tutor: {'classes': [], 'total': 0, 'courses': set()} for tutor in self.tutors}
        tutor_course_diversity = {}
        
        for (tutor, course, class_id), var in self.x_vars.items():
            if var.varValue and var.varValue > 0.5:
                assignments[(course, class_id)] = tutor
                tutor_loads[tutor]['classes'].append((course, class_id))
                tutor_loads[tutor]['total'] += 1
                tutor_loads[tutor]['courses'].add(course)
        
        for tutor in self.tutors:
            tutor_course_diversity[tutor] = len(tutor_loads[tutor]['courses'])
        
        unassigned_classes = []
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
    
    pattern = r'\b(ACTL|RISK|COMM)\d{4}(?:/\d{4})?\b'
    codes = [match.group() for match in re.finditer(pattern, str(text))]
    
    return list(set(codes))


def extract_degree(text: str) -> str:
    """Extract degree information from text."""
    if pd.isna(text) or str(text).strip() == '':
        return "Not Specified"
    
    return str(text).strip()


def normalize_degree(degree_text: str) -> str:
    """
    Normalize degree text to standard categories.
    Returns: 'PhD', 'Master', or 'Bachelor' (default)
    """
    degree_upper = degree_text.upper()
    
    if 'PHD' in degree_upper or 'PH.D' in degree_upper or 'DOCTOR' in degree_upper or 'PH D' in degree_upper or 'DOCTORATE' in degree_upper:
        return 'PhD'
    elif 'MASTER' in degree_upper or 'MSC' in degree_upper or 'MA' in degree_upper or 'M.SC' in degree_upper or 'M.A' in degree_upper or 'M SC' in degree_upper:
        return 'Master'
    elif 'BACHELOR' in degree_upper or 'BSC' in degree_upper or 'BA' in degree_upper or 'B.SC' in degree_upper or 'B.A' in degree_upper or 'B SC' in degree_upper or 'UNDERGRAD' in degree_upper:
        return 'Bachelor'
    else:
        # Default to Bachelor if unclear
        return 'Bachelor'


def parse_max_classes(value) -> Tuple[int, str]:
    """Parse max_classes value from various formats."""
    if pd.isna(value) or str(value).strip() == '':
        return 3, "Empty (defaulted to 3)"
    
    value_str = str(value).strip()
    
    if isinstance(value, pd.Timestamp) or 'Timestamp' in str(type(value)):
        return 3, f"Date value '{value}' (defaulted to 3)"
    
    if re.search(r'\b(19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b', value_str):
        return 3, f"Date value '{value_str}' (defaulted to 3)"
    
    try:
        num = int(float(value_str))
        if num > 100:
            return 3, f"Invalid large number '{value_str}' (defaulted to 3)"
        if num > 50:
            return 3, f"Value too high '{num}' (defaulted to 3)"
        return num, "OK"
    except:
        pass
    
    if '-' in value_str and not re.search(r'\d{4}', value_str):
        try:
            parts = value_str.split('-')
            low = int(parts[0].strip())
            high = int(parts[1].strip())
            if low <= 50 and high <= 50:
                avg = (low + high) // 2
                return avg, f"Range {value_str} (using average: {avg})"
        except:
            pass
    
    if '+' in value_str:
        try:
            num = int(value_str.replace('+', '').strip())
            if num <= 50:
                return num, f"'{value_str}' (using {num})"
        except:
            pass
    
    numbers = re.findall(r'\b\d+\b', value_str)
    for num_str in numbers:
        num = int(num_str)
        if 1 <= num <= 50:
            return num, f"Extracted {num} from '{value_str}'"
    
    return 3, f"Could not parse '{value_str}' (defaulted to 3)"


def classify_course_level(course_code: str) -> str:
    """Classify course as PG or UG. Default: PG."""
    return 'PG'


def load_file1_classes(file_path: str) -> pd.DataFrame:
    """Load and parse File 1 (Classes for T3)."""
    df = pd.read_excel(file_path, header=None)
    
    classes_data = []
    current_course = None
    
    for idx, row in df.iterrows():
        val = str(row[0]).strip()
        
        if val.startswith('ACTL') or val.startswith('RISK') or val.startswith('COMM'):
            if pd.isna(row[1]) or 'Class' not in str(row[1]):
                current_course = val
                continue
        
        if current_course and pd.notna(row[0]) and pd.notna(row[1]):
            class_type = str(row[1]).strip()
            
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
                    'course_level': classify_course_level(current_course)
                })
    
    return pd.DataFrame(classes_data)


def load_file2_tutors(file_path: str) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, int], Dict[str, str], Dict[str, str]]:
    """Load and parse File 2 (Tutor Preferences)."""
    df = pd.read_excel(file_path, sheet_name=0)
    
    first_name_col = df.columns[6]
    last_name_col = df.columns[8]
    pref_name_col = df.columns[10]
    degree_col = df.columns[57]
    t3_pref_col = df.columns[108]
    max_classes_col = df.columns[110]
    
    tutors_data = []
    preferences = {}
    max_classes = {}
    parsing_status = {}
    degrees = {}
    
    for idx, row in df.iterrows():
        pref_name = row[pref_name_col]
        if pd.isna(pref_name) or str(pref_name).strip() == '':
            first = str(row[first_name_col]).strip() if pd.notna(row[first_name_col]) else ''
            last = str(row[last_name_col]).strip() if pd.notna(row[last_name_col]) else ''
            tutor_name = f"{first} {last}".strip()
        else:
            tutor_name = str(pref_name).strip()
        
        if not tutor_name or tutor_name == '':
            continue
        
        degree_info = extract_degree(row[degree_col])
        
        t3_pref_text = row[t3_pref_col]
        t3_courses_raw = extract_course_codes(t3_pref_text)
        
        t3_courses_expanded = []
        for code in t3_courses_raw:
            if '/' in code:
                parts = code.split('/')
                base = parts[0]
                prefix = base[:4]
                second_code = prefix + parts[1]
                t3_courses_expanded.append(base)
                t3_courses_expanded.append(second_code)
            else:
                t3_courses_expanded.append(code)
        
        t3_courses_expanded = list(set(t3_courses_expanded))
        
        if len(t3_courses_expanded) > 0:
            max_val, status = parse_max_classes(row[max_classes_col])
            
            tutors_data.append({
                'tutor_name': tutor_name,
                'first_name': str(row[first_name_col]) if pd.notna(row[first_name_col]) else '',
                'last_name': str(row[last_name_col]) if pd.notna(row[last_name_col]) else '',
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
        page_title="Tutor Assignment Optimizer (Dual Preferences)",
        page_icon="👨‍🏫",
        layout="wide"
    )
    
    st.title("👨‍🏫 Tutor-Class Assignment Optimizer (T3) - Dual Preferences")
    st.markdown("**Linear Programming with Tutor Preferences + Academic Preferences + Time Conflicts + Degree Requirements**")
    st.markdown("---")
    
    if 'step' not in st.session_state:
        st.session_state.step = 1
    
    if st.session_state.step == 1:
        show_upload_step()
    elif st.session_state.step == 2:
        show_course_level_step()
    elif st.session_state.step == 3:
        show_review_tutors_step()
    elif st.session_state.step == 4:
        show_academic_preferences_step()
    elif st.session_state.step == 5:
        show_analysis_step()
    elif st.session_state.step == 6:
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
                classes_df = load_file1_classes(file1)
                st.session_state.classes_df = classes_df
                
                tutors_df, preferences, max_classes, parsing_status, degrees = load_file2_tutors(file2)
                st.session_state.tutors_df = tutors_df
                st.session_state.tutor_preferences = preferences
                st.session_state.max_classes = max_classes
                st.session_state.parsing_status = parsing_status
                st.session_state.degrees = degrees
                
                st.success("✅ Files loaded successfully!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Classes Found (TUT/LAB)", len(classes_df))
                with col2:
                    st.metric("Unique Courses", classes_df['course'].nunique())
                with col3:
                    st.metric("Tutors with T3 Preferences", len(tutors_df))
                
                st.subheader("🎓 Tutor Degree Distribution")
                degree_counts = {}
                for degree in degrees.values():
                    normalized = normalize_degree(degree)
                    degree_counts[normalized] = degree_counts.get(normalized, 0) + 1
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("PhD Tutors", degree_counts.get('PhD', 0))
                    st.caption("Can teach PG & UG")
                with col2:
                    st.metric("Master Tutors", degree_counts.get('Master', 0))
                    st.caption("Can teach UG only")
                with col3:
                    st.metric("Bachelor Tutors", degree_counts.get('Bachelor', 0))
                    st.caption("Can teach UG only")
                
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
    
    unique_courses = sorted(classes_df['course'].unique())
    
    st.subheader("Set Course Levels")
    
    if 'course_levels' not in st.session_state:
        st.session_state.course_levels = {course: 'PG' for course in unique_courses}
    
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
            
            num_classes = len(classes_df[classes_df['course'] == course])
            st.caption(f"Classes: {num_classes}")
            st.markdown("---")
    
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
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("← Back to Upload"):
            st.session_state.step = 1
            st.rerun()
    
    with col2:
        if st.button("Next: Review Tutors →", type="primary"):
            classes_df['course_level'] = classes_df['course'].map(st.session_state.course_levels)
            st.session_state.classes_df = classes_df
            st.session_state.step = 3
            st.rerun()


def show_review_tutors_step():
    """Step 3: Review and edit tutor information."""
    st.header("Step 3: Review & Edit Tutor Information")
    
    tutors_df = st.session_state.tutors_df
    parsing_status = st.session_state.parsing_status
    degrees = st.session_state.degrees
    
    st.markdown("""
    **Review and edit tutor information below:**
    - 🎓 **Education Level**: PhD tutors can teach PG & UG; Non-PhD can only teach UG
    - 📊 **Max Classes**: Maximum number of classes per tutor
    
    **Status Indicators:**
    - 🟢 **Green (✅)**: Information is clear and valid
    - 🟡 **Yellow (⚠️)**: Information needs attention
    - 🔴 **Red (❌)**: No information provided
    - 🟠 **Orange (⚠️ Check Load)**: Max classes > 4, verify if this workload is appropriate
    """)
    
    st.markdown("---")
    
    # Check for issues
    needs_review_max = tutors_df[tutors_df['max_classes_status'].str.contains('defaulted|Could not parse', case=False, na=False)]
    high_load_tutors = tutors_df[tutors_df['max_classes'] > 4]
    
    if len(needs_review_max) > 0:
        st.warning(f"⚠️ {len(needs_review_max)} tutor(s) have max_classes values that may need correction")
    else:
        st.success("✅ All max_classes values parsed successfully!")
    
    if len(high_load_tutors) > 0:
        st.warning(f"🟠 {len(high_load_tutors)} tutor(s) have max classes > 4 - please verify these workloads are appropriate")
    
    st.markdown("---")
    
    # Editable table
    st.subheader("Edit Tutor Information")
    
    # Header row
    col1, col2, col3, col4, col5 = st.columns([3, 3, 2, 3, 1])
    with col1:
        st.markdown("**Tutor Name**")
    with col2:
        st.markdown("**Original Degree Info**")
    with col3:
        st.markdown("**Qualification**")
    with col4:
        st.markdown("**Max Classes Status**")
    with col5:
        st.markdown("**Max**")
    
    st.markdown("---")
    
    edited_degrees = {}
    edited_max_classes = {}
    
    # Data rows
    for idx, row in tutors_df.iterrows():
        col1, col2, col3, col4, col5 = st.columns([3, 3, 2, 3, 1])
        
        with col1:
            st.markdown(f"**{row['tutor_name']}**")
        
        with col2:
            # Show original degree text with color-coded status
            original_degree = row['degree']
            
            # Determine status color based on original text
            if pd.isna(original_degree) or str(original_degree).strip() == '' or original_degree == 'Not Specified':
                # Red - No information
                st.error(f"❌ No degree information")
                degree_status = "red"
                default_degree = 'Bachelor'
            else:
                # Check if we can clearly identify the degree
                normalized = normalize_degree(original_degree)
                if normalized in ['PhD', 'Master', 'Bachelor']:
                    # Green - OK
                    st.success(f"✅ {original_degree}")
                    degree_status = "green"
                    default_degree = normalized
                else:
                    # Yellow - Needs attention (shouldn't happen now, but keep as fallback)
                    st.warning(f"⚠️ {original_degree}")
                    degree_status = "yellow"
                    default_degree = 'Bachelor'
        
        with col3:
            # Degree dropdown - Only PhD, Master, Bachelor
            degree_options = ['PhD', 'Master', 'Bachelor']
            
            # Get current degree or use default
            current_degree = degrees.get(row['tutor_name'], 'Bachelor')
            normalized_degree = normalize_degree(current_degree)
            
            # Ensure normalized degree is in options
            if normalized_degree not in degree_options:
                normalized_degree = default_degree
            
            default_idx = degree_options.index(normalized_degree)
            
            new_degree = st.selectbox(
                "Qualification",
                options=degree_options,
                index=default_idx,
                key=f"degree_{idx}",
                label_visibility="collapsed"
            )
            
            # Show teaching eligibility
            if new_degree == 'PhD':
                st.caption("✅ PG & UG")
            else:
                st.caption("⚠️ UG only")
            
            edited_degrees[row['tutor_name']] = new_degree
        
        with col4:
            # Max classes status with color coding
            status = row['max_classes_status']
            current_max_value = int(row['max_classes'])
            
            # Check if max classes > 4
            if current_max_value > 4:
                st.warning(f"🟠 Check Load: {current_max_value} classes")
                st.caption(f"Original: {status[:30]}")
            elif 'defaulted' in status.lower() or 'could not parse' in status.lower():
                st.error(f"❌ {status[:50]}")
            elif 'range' in status.lower() or 'extracted' in status.lower():
                st.warning(f"⚠️ {status[:50]}")
            else:
                st.success(f"✅ {status[:50]}")
        
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
            
            # Show warning icon if > 5
            if new_max > 5:
                st.caption("⚠️ High")
        
        st.markdown("---")
    
    # Summary of education levels
    st.subheader("📊 Summary of Education Levels")
    
    degree_summary = {}
    for degree in edited_degrees.values():
        degree_summary[degree] = degree_summary.get(degree, 0) + 1
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("PhD", degree_summary.get('PhD', 0))
        st.caption("Can teach PG & UG")
    with col2:
        st.metric("Master", degree_summary.get('Master', 0))
        st.caption("Can teach UG only")
    with col3:
        st.metric("Bachelor", degree_summary.get('Bachelor', 0))
        st.caption("Can teach UG only")
    
    # Summary of workload warnings
    st.subheader("⚠️ Workload Summary")
    
    high_load_count = sum(1 for max_val in edited_max_classes.values() if max_val > 4)
    if high_load_count > 0:
        st.warning(f"🟠 {high_load_count} tutor(s) with max classes > 4:")
        high_load_list = [(name, max_val) for name, max_val in edited_max_classes.items() if max_val > 4]
        high_load_list.sort(key=lambda x: x[1], reverse=True)
        
        for name, max_val in high_load_list:
            st.write(f"- **{name}**: {max_val} classes")
    else:
        st.success("✅ All tutors have reasonable workloads (≤ 4 classes)")
    
    st.markdown("---")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Back to Course Levels"):
            st.session_state.step = 2
            st.rerun()
    
    with col3:
        if st.button("Next: Academic Preferences →", type="primary"):
            st.session_state.degrees = edited_degrees
            st.session_state.max_classes = edited_max_classes
            st.session_state.step = 4
            st.rerun()


def show_academic_preferences_step():
    """Step 4: Set academic preferences (a_tc ratings)."""
    st.header("Step 4: Set Academic Preferences")
    
    classes_df = st.session_state.classes_df
    tutors_df = st.session_state.tutors_df
    tutor_preferences = st.session_state.tutor_preferences
    degrees = st.session_state.degrees
    
    st.markdown("""
    **Academic Preferences (a_tc)** are ratings (1-10) from course coordinators indicating how suitable each tutor is for each course.
    
    **Rating Scale:**
    - **10**: Strongly preferred (excellent fit, high expertise)
    - **7-9**: Good fit (recommended)
    - **5-6**: Acceptable (neutral, adequate)
    - **3-4**: Less preferred (concerns, but acceptable if needed)
    - **1-2**: Avoid (significant concerns, last resort only)
    - **Default**: 5 (neutral) if not specified
    
    **Academic Veto Threshold (θ):** Tutors with ratings ≤ θ will be blocked from teaching that course.
    """)
    
    st.markdown("---")
    
    # Initialize academic preferences if not exists
    if 'academic_preferences' not in st.session_state:
        st.session_state.academic_preferences = {}
    
    # Get unique courses
    courses = sorted(classes_df['course'].unique())
    
    # Option to upload academic preferences file or enter manually
    st.subheader("📥 Option 1: Upload Academic Preferences (Optional)")
    
    st.markdown("""
    **Upload a CSV/Excel file with columns:**
    - `Tutor`: Tutor name
    - `Course`: Course code
    - `Rating`: Academic preference rating (1-10)
    """)
    
    uploaded_prefs = st.file_uploader(
        "Upload Academic Preferences File (Optional)",
        type=['csv', 'xlsx', 'xls'],
        key='academic_prefs_file'
    )
    
    if uploaded_prefs is not None:
        try:
            if uploaded_prefs.name.endswith('.csv'):
                prefs_df = pd.read_csv(uploaded_prefs)
            else:
                prefs_df = pd.read_excel(uploaded_prefs)
            
            # Validate columns
            required_cols = ['Tutor', 'Course', 'Rating']
            if all(col in prefs_df.columns for col in required_cols):
                # Load preferences
                for idx, row in prefs_df.iterrows():
                    tutor = str(row['Tutor']).strip()
                    course = str(row['Course']).strip()
                    rating = float(row['Rating'])
                    
                    # Validate rating
                    if 1 <= rating <= 10:
                        st.session_state.academic_preferences[(tutor, course)] = rating
                
                st.success(f"✅ Loaded {len(st.session_state.academic_preferences)} academic preferences from file!")
            else:
                st.error(f"❌ File must contain columns: {', '.join(required_cols)}")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    st.markdown("---")
    
    # Option to set preferences manually
    st.subheader("📝 Option 2: Set Academic Preferences Manually")
    
    st.info("💡 You can set ratings for specific tutor-course pairs below. Unspecified pairs default to 5 (neutral).")
    
    # Filter by course to make it manageable
    selected_course = st.selectbox(
        "Select Course to Set Preferences:",
        options=['-- Select a course --'] + courses,
        key='pref_course_select'
    )
    
    if selected_course != '-- Select a course --':
        st.markdown(f"### Set preferences for **{selected_course}**")
        
        # Get tutors who prefer this course
        course_level = classes_df[classes_df['course'] == selected_course].iloc[0]['course_level']
        
        # Filter eligible tutors
        eligible_tutors = []
        for tutor in tutors_df['tutor_name'].unique():
            # Check if tutor prefers this course
            if selected_course in tutor_preferences.get(tutor, []):
                # Check degree eligibility
                tutor_degree = degrees.get(tutor, 'Bachelor')
                if tutor_degree == 'PhD' or course_level == 'UG':
                    eligible_tutors.append(tutor)
        
        eligible_tutors = sorted(eligible_tutors)
        
        if len(eligible_tutors) == 0:
            st.warning(f"⚠️ No eligible tutors found for {selected_course}. Either no tutors prefer this course, or (if PG) no PhD tutors are available.")
        else:
            st.markdown(f"**{len(eligible_tutors)} eligible tutor(s) for {selected_course}:**")
            
            # Create a form to prevent constant reruns
            with st.form(f"prefs_form_{selected_course}"):
                new_ratings = {}
                
                # Display tutors in columns
                num_cols = 3
                cols = st.columns(num_cols)
                
                for idx, tutor in enumerate(eligible_tutors):
                    with cols[idx % num_cols]:
                        current_rating = st.session_state.academic_preferences.get((tutor, selected_course), 5.0)
                        
                        new_rating = st.slider(
                            f"{tutor}",
                            min_value=1.0,
                            max_value=10.0,
                            value=current_rating,
                            step=0.5,
                            key=f"rating_{tutor}_{selected_course}"
                        )
                        
                        # Color-code based on rating
                        if new_rating >= 7:
                            st.success(f"✅ {new_rating} - Recommended")
                        elif new_rating >= 5:
                            st.info(f"ℹ️ {new_rating} - Acceptable")
                        elif new_rating >= 3:
                            st.warning(f"⚠️ {new_rating} - Less preferred")
                        else:
                            st.error(f"❌ {new_rating} - Avoid")
                        
                        new_ratings[tutor] = new_rating
                        
                        st.markdown("---")
                
                # Submit button
                if st.form_submit_button("💾 Save Preferences for this Course"):
                    for tutor, rating in new_ratings.items():
                        st.session_state.academic_preferences[(tutor, selected_course)] = rating
                    st.success(f"✅ Saved preferences for {selected_course}!")
    
    st.markdown("---")
    
    # Summary of academic preferences
    st.subheader("📊 Academic Preferences Summary")
    
    num_set = len(st.session_state.academic_preferences)
    st.metric("Preferences Set", num_set)
    
    if num_set > 0:
        with st.expander("📋 View All Academic Preferences"):
            prefs_display = []
            for (tutor, course), rating in sorted(st.session_state.academic_preferences.items()):
                prefs_display.append({
                    'Tutor': tutor,
                    'Course': course,
                    'Rating': rating
                })
            
            prefs_display_df = pd.DataFrame(prefs_display)
            st.dataframe(prefs_display_df, use_container_width=True, hide_index=True)
            
            # Download template
            from io import BytesIO
            output = BytesIO()
            prefs_display_df.to_csv(output, index=False)
            
            st.download_button(
                label="📥 Download Current Preferences (CSV)",
                data=output.getvalue(),
                file_name="academic_preferences.csv",
                mime="text/csv"
            )
    
    st.markdown("---")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Back to Review Tutors"):
            st.session_state.step = 3
            st.rerun()
    
    with col3:
        if st.button("Next: Analysis →", type="primary"):
            st.session_state.step = 5
            st.rerun()


def show_analysis_step():
    """Step 5: Show data analysis and visualizations."""
    st.header("Step 5: Data Analysis")
    
    classes_df = st.session_state.classes_df
    tutors_df = st.session_state.tutors_df
    tutor_preferences = st.session_state.tutor_preferences
    academic_preferences = st.session_state.academic_preferences
    max_classes = st.session_state.max_classes
    degrees = st.session_state.degrees
    
    st.subheader("📊 Overview")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Classes", len(classes_df))
    with col2:
        st.metric("Unique Courses", classes_df['course'].nunique())
    with col3:
        st.metric("Available Tutors", len(tutors_df))
    with col4:
        total_capacity = sum(max_classes.values())
        st.metric("Total Capacity", total_capacity)
    with col5:
        phd_count = sum(1 for d in degrees.values() if d == 'PhD')
        st.metric("PhD Tutors", phd_count)
    with col6:
        num_acad_prefs = len(academic_preferences)
        st.metric("Academic Prefs Set", num_acad_prefs)
    
    if total_capacity < len(classes_df):
        st.warning(f"⚠️ Total tutor capacity ({total_capacity}) is less than total classes ({len(classes_df)}). Optimization will proceed and show unassigned classes.")
    else:
        surplus = total_capacity - len(classes_df)
        st.success(f"✅ Sufficient capacity: {surplus} extra class slots available")
    
    st.markdown("---")
    
    # Dual preference analysis
    st.subheader("🎯 Dual Preference Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Tutor Preferences (p_tc)**")
        total_tutor_prefs = sum(len(courses) for courses in tutor_preferences.values())
        st.metric("Total Tutor-Course Pairs", total_tutor_prefs)
        st.caption("Binary indicator: Tutor wants to teach course (p_tc = 10)")
    
    with col2:
        st.markdown("**Academic Preferences (a_tc)**")
        st.metric("Preferences Set", len(academic_preferences))
        st.caption("Ratings 1-10 from coordinators (default: 5)")
        
        if len(academic_preferences) > 0:
            avg_rating = np.mean(list(academic_preferences.values()))
            st.metric("Average Rating", f"{avg_rating:.2f}")
    
    # Academic preference distribution
    if len(academic_preferences) > 0:
        st.subheader("📊 Academic Preference Distribution")
        
        ratings = list(academic_preferences.values())
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Count by rating range
            excellent = sum(1 for r in ratings if r >= 9)
            good = sum(1 for r in ratings if 7 <= r < 9)
            acceptable = sum(1 for r in ratings if 5 <= r < 7)
            less_preferred = sum(1 for r in ratings if 3 <= r < 5)
            avoid = sum(1 for r in ratings if r < 3)
            
            st.write("**Rating Distribution:**")
            st.write(f"- Excellent (9-10): {excellent}")
            st.write(f"- Good (7-9): {good}")
            st.write(f"- Acceptable (5-7): {acceptable}")
            st.write(f"- Less Preferred (3-5): {less_preferred}")
            st.write(f"- Avoid (1-3): {avoid}")
        
        with col2:
            fig_ratings = px.histogram(
                x=ratings,
                nbins=20,
                title='Academic Preference Ratings Distribution',
                labels={'x': 'Rating', 'y': 'Count'},
                color_discrete_sequence=['#4ECDC4']
            )
            fig_ratings.update_layout(showlegend=False)
            st.plotly_chart(fig_ratings, use_container_width=True)
    
    st.markdown("---")
    
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
    
    st.subheader("📋 Course Coverage Analysis (with Degree & Academic Veto)")
    
    st.info("ℹ️ Even if some courses show insufficient capacity, optimization will still run and show which classes remain unassigned.")
    
    # Use default theta for analysis
    default_theta = 2.0
    
    coverage_data = []
    courses = sorted(classes_df['course'].unique())
    
    for course in courses:
        course_df = classes_df[classes_df['course'] == course]
        num_classes = len(course_df)
        course_level = course_df.iloc[0]['course_level']
        
        # Find qualified tutors considering both preferences and academic veto
        if course_level == 'PG':
            qualified_tutors = [
                tutor for tutor in tutors_df['tutor_name'].unique()
                if course in tutor_preferences.get(tutor, []) 
                and degrees.get(tutor, '') == 'PhD'
                and academic_preferences.get((tutor, course), 5.0) > default_theta
            ]
        else:
            qualified_tutors = [
                tutor for tutor in tutors_df['tutor_name'].unique()
                if course in tutor_preferences.get(tutor, [])
                and academic_preferences.get((tutor, course), 5.0) > default_theta
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
    
    problematic = coverage_df[coverage_df['Status'] == '⚠️']
    if len(problematic) > 0:
        st.warning(f"⚠️ {len(problematic)} course(s) have insufficient qualified tutor capacity:")
        st.dataframe(problematic, use_container_width=True, hide_index=True)
        
        for idx, row in problematic.iterrows():
            if row['Level'] == 'PG' and row['Qualified Tutors'] == 0:
                st.error(f"**{row['Course']}**: No PhD tutors available who prefer this course and pass academic veto!")
            elif row['Qualified Tutors'] == 0:
                st.error(f"**{row['Course']}**: No tutors have expressed preference for this course or pass academic veto!")
        
        st.info("💡 The optimization will still run and show which specific classes cannot be assigned.")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to Academic Preferences"):
            st.session_state.step = 4
            st.rerun()
    
    with col2:
        if st.button("Run Optimization →", type="primary"):
            st.session_state.step = 6
            st.rerun()


def show_optimization_step():
    """Step 6: Run optimization and show results - WITH FORM TO PREVENT RERUNS."""
    st.header("Step 6: Optimization Results")
    
    classes_df = st.session_state.classes_df
    tutors_df = st.session_state.tutors_df
    tutor_preferences = st.session_state.tutor_preferences
    academic_preferences = st.session_state.academic_preferences
    max_classes = st.session_state.max_classes
    degrees = st.session_state.degrees
    
    # Use a form to prevent reruns on input changes
    with st.form("optimization_form"):
        st.subheader("⚙️ Optimization Parameters")
        
        st.markdown("### Preference Weights")
        col1, col2 = st.columns(2)
        
        with col1:
            w1 = st.slider(
                "Tutor Preference Weight",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=1.0,
                help="Weight for tutor binary preference (p_tc = 0 or 10)"
            )
            
            st.info(f"""
            **Current: {w1}**
            
            Multiplier for tutor preferences
            - Higher → Prioritize tutor choices
            """)
        
        with col2:
            w2 = st.slider(
                "Academic Preference Weight",
                min_value=0.0,
                max_value=50.0,
                value=20.0,
                step=1.0,
                help="Weight for academic ratings (a_tc = 1-10)"
            )
            
            st.info(f"""
            **Current: {w2}**
            
            Multiplier for academic ratings
            - Higher → Prioritize coordinator preferences
            """)
        
        st.markdown("---")
        
        st.markdown("### Academic Veto Threshold")
        
        theta = st.slider(
            "Academic Veto Threshold",
            min_value=0.0,
            max_value=4.0,
            value=2.0,
            step=0.5,
            help="Tutors with rating ≤ θ are blocked from course. Set to 0 to disable veto."
        )
        
        st.warning(f"""
        **Current Threshold: {theta}**
        
        - Tutors with a_tc ≤ {theta} will be **blocked** from teaching that course
        - Set to 0 to disable academic veto
        """)
        
        st.markdown("---")
        
        st.markdown("### Degree Priority & Course Diversity")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            course_diversity_penalty = st.slider(
                "Course Diversity Penalty",
                min_value=0.0,
                max_value=20.0,
                value=20.0,
                step=0.5,
                help="Higher penalty encourages tutors to teach fewer different courses"
            )
            
            st.info(f"""
            **Current: {course_diversity_penalty}**
            
            - **0**: No penalty
            - **5**: Moderate
            - **10+**: Strong penalty
            """)
        
        with col2:
            phd_priority_bonus = st.slider(
                "PhD Priority Bonus",
                min_value=0.0,
                max_value=20.0,
                value=20.0,
                step=0.5,
                help="Higher bonus gives more priority to PhD students"
            )
            
            st.success(f"""
            **Current: {phd_priority_bonus}**
            
            - **0**: No priority
            - **10**: Strong 
            - Must be > γ
            """)
        
        with col3:
            master_priority_bonus = st.slider(
                "Master Priority Bonus",
                min_value=0.0,
                max_value=15.0,
                value=15.0,
                step=0.5,
                help="Higher bonus gives more priority to Master students (should be < β)"
            )
            
            st.info(f"""
            **Current: {master_priority_bonus}**
            
            - **0**: No priority
            - **5**: Medium 
            - Must be < β
            """)
        
        # Validate constraint β > γ
        if phd_priority_bonus <= master_priority_bonus:
            st.error("⚠️ Constraint violation: β (PhD bonus) must be > γ (Master bonus)")
        
        # Submit button
        run_optimization = st.form_submit_button(
            "🚀 Run Optimization", 
            type="primary",
            disabled=(phd_priority_bonus <= master_priority_bonus)
        )
    
    # Only run optimization when button is clicked
    if run_optimization or 'optimization_results' in st.session_state:
        
        if run_optimization:
            # Clear previous results
            if 'optimization_results' in st.session_state:
                del st.session_state.optimization_results
            
            st.markdown("---")
            
            # Build preference dictionaries
            tutor_pref_dict = {}
            for tutor, courses in tutor_preferences.items():
                for course in courses:
                    tutor_pref_dict[(tutor, course)] = 10  # Binary: 10 if preferred, 0 otherwise
            
            # academic_preferences already in correct format
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Initializing optimization model with dual preferences...")
                progress_bar.progress(20)
                
                lp = TutorAssignmentLP(
                    classes_df=classes_df,
                    tutors_df=tutors_df,
                    tutor_preferences=tutor_pref_dict,
                    academic_preferences=academic_preferences,
                    tutor_max_classes=max_classes,
                    tutor_degrees=degrees,
                    w1=w1,
                    w2=w2,
                    theta=theta,
                    course_diversity_penalty=course_diversity_penalty,
                    phd_priority_bonus=phd_priority_bonus,
                    master_priority_bonus=master_priority_bonus
                )
                
                status_text.text("Building model...")
                progress_bar.progress(40)
                lp.build_model()
                
                num_vars = len(lp.x_vars) + len(lp.y_vars)
                st.info(f"📊 Model created with {num_vars} decision variables ({len(lp.x_vars)} assignments + {len(lp.y_vars)} course indicators)")
                
                # Show eligibility stats
                total_possible = len(tutors_df) * len(classes_df)
                eligible_pct = (len(lp.eligible_assignments) / total_possible) * 100 if total_possible > 0 else 0
                st.info(f"✅ {len(lp.eligible_assignments)} eligible assignments ({eligible_pct:.1f}% of all possible combinations)")
                
                status_text.text("Solving optimization problem... (no time limit - finding optimal solution)")
                progress_bar.progress(60)
                
                # Call solve without time limit (will run until optimal)
                solution = lp.solve(time_limit=None)
                
                progress_bar.progress(100)
                status_text.text(f"✅ Optimization completed in {solution.get('solve_time', 0):.1f} seconds")
                
                # Store results in session state
                st.session_state.optimization_results = solution
                st.session_state.optimization_params = {
                    'w1': w1,
                    'w2': w2,
                    'theta': theta,
                    'alpha': course_diversity_penalty,
                    'beta': phd_priority_bonus,
                    'gamma': master_priority_bonus
                }
                
            except Exception as e:
                progress_bar.progress(100)
                status_text.text("❌ Error occurred")
                st.error(f"❌ Error during optimization: {str(e)}")
                st.exception(e)
                return
        
        # Display results from session state
        solution = st.session_state.optimization_results
        params = st.session_state.get('optimization_params', {})
        
        st.markdown("---")
        
        # Display parameters used
        with st.expander("⚙️ Parameters Used"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**w₁**: {params.get('w1', 'N/A')}")
                st.write(f"**w₂**: {params.get('w2', 'N/A')}")
            with col2:
                st.write(f"**θ**: {params.get('theta', 'N/A')}")
                st.write(f"**α**: {params.get('alpha', 'N/A')}")
            with col3:
                st.write(f"**β**: {params.get('beta', 'N/A')}")
                st.write(f"**γ**: {params.get('gamma', 'N/A')}")
        
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
        
        if solution['status'] in ['Optimal', 'Not Solved']:
            if unassigned_count == 0:
                st.success("✅ All classes assigned!")
            else:
                st.warning(f"⚠️ {unassigned_count} classes could not be assigned (see details below).")
            
            # Create results dataframe
            results_data = []
            for idx, row in classes_df.iterrows():
                course = row['course']
                class_id = row['class_id']
                assigned_tutor = solution['assignments'].get((course, class_id), 'UNASSIGNED')
                
                tutor_degree = degrees.get(assigned_tutor, 'N/A') if assigned_tutor != 'UNASSIGNED' else 'N/A'
                
                # Get preferences
                if assigned_tutor != 'UNASSIGNED':
                    tutor_pref = tutor_preferences.get(assigned_tutor, [])
                    has_tutor_pref = 'Yes' if course in tutor_pref else 'No'
                    acad_pref = academic_preferences.get((assigned_tutor, course), 5.0)
                else:
                    has_tutor_pref = 'N/A'
                    acad_pref = 'N/A'
                
                results_data.append({
                    'Course': course,
                    'Level': row['course_level'],
                    'Class ID': class_id,
                    'Type': row['type'],
                    'Section': row['section'],
                    'Time': row['time'],
                    'Assigned Tutor': assigned_tutor,
                    'Tutor Degree': tutor_degree,
                    'Tutor Pref': has_tutor_pref,
                    'Academic Rating': acad_pref
                })
            
            results_df = pd.DataFrame(results_data)
            
            # Show assignment table WITH FORM to prevent reruns
            st.subheader("📋 Class Assignments")
            
            with st.form("filter_form"):
                col1, col2, col3, col4 = st.columns([3, 3, 2, 2])
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
                with col4:
                    apply_filter = st.form_submit_button("Apply Filters")
            
            # Apply filters
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
                
                courses_count = {}
                pg_count = 0
                ug_count = 0
                total_acad_rating = 0
                
                for course, class_id in assigned_classes:
                    courses_count[course] = courses_count.get(course, 0) + 1
                    class_level = classes_df[
                        (classes_df['course'] == course) & 
                        (classes_df['class_id'] == class_id)
                    ].iloc[0]['course_level']
                    if class_level == 'PG':
                        pg_count += 1
                    else:
                        ug_count += 1
                    
                    # Get academic rating
                    acad_rating = academic_preferences.get((tutor, course), 5.0)
                    total_acad_rating += acad_rating
                
                avg_acad_rating = total_acad_rating / len(assigned_classes) if len(assigned_classes) > 0 else 0
                
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
                    'Avg Academic Rating': f"{avg_acad_rating:.2f}" if load['total'] > 0 else "N/A",
                    'Courses Assigned': courses_str if courses_str else 'None'
                })
            
            workload_df = pd.DataFrame(workload_data)
            workload_df = workload_df.sort_values('Total Classes', ascending=False)
            st.dataframe(workload_df, use_container_width=True, hide_index=True)
            
            # Degree Priority Analysis
            st.subheader("🎓 Degree Priority Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            # PhD Analysis
            with col1:
                st.markdown("**PhD Priority (Highest)**")
                phd_tutors = workload_df[workload_df['Degree'] == 'PhD']
                phd_assigned = phd_tutors[phd_tutors['Total Classes'] > 0].shape[0]
                phd_total = len(phd_tutors)
                st.metric("PhD Tutors Assigned", f"{phd_assigned}/{phd_total}")
                if phd_total > 0:
                    st.caption(f"{phd_assigned/phd_total*100:.0f}% utilization")
                
                phd_classes = phd_tutors['Total Classes'].sum()
                st.metric("Classes by PhD", phd_classes)
            
            # Master Analysis
            with col2:
                st.markdown("**Master Priority (Medium)**")
                master_tutors = workload_df[workload_df['Degree'] == 'Master']
                master_assigned = master_tutors[master_tutors['Total Classes'] > 0].shape[0]
                master_total = len(master_tutors)
                st.metric("Master Tutors Assigned", f"{master_assigned}/{master_total}")
                if master_total > 0:
                    st.caption(f"{master_assigned/master_total*100:.0f}% utilization")
                
                master_classes = master_tutors['Total Classes'].sum()
                st.metric("Classes by Master", master_classes)
            
            # Bachelor Analysis
            with col3:
                st.markdown("**Bachelor Priority (Base)**")
                bachelor_tutors = workload_df[workload_df['Degree'] == 'Bachelor']
                bachelor_assigned = bachelor_tutors[bachelor_tutors['Total Classes'] > 0].shape[0]
                bachelor_total = len(bachelor_tutors)
                st.metric("Bachelor Tutors Assigned", f"{bachelor_assigned}/{bachelor_total}")
                if bachelor_total > 0:
                    st.caption(f"{bachelor_assigned/bachelor_total*100:.0f}% utilization")
                
                bachelor_classes = bachelor_tutors['Total Classes'].sum()
                st.metric("Classes by Bachelor", bachelor_classes)
            
            # Preference satisfaction analysis
            st.subheader("🎯 Preference Satisfaction Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Tutor Preferences**")
                assigned_with_tutor_pref = results_df[(results_df['Assigned Tutor'] != 'UNASSIGNED') & (results_df['Tutor Pref'] == 'Yes')]
                assigned_without_tutor_pref = results_df[(results_df['Assigned Tutor'] != 'UNASSIGNED') & (results_df['Tutor Pref'] == 'No')]
                
                total_assigned = len(results_df[results_df['Assigned Tutor'] != 'UNASSIGNED'])
                
                if total_assigned > 0:
                    st.metric("Classes with Tutor Preference", 
                             f"{len(assigned_with_tutor_pref)}/{total_assigned}",
                             delta=f"{len(assigned_with_tutor_pref)/total_assigned*100:.1f}%")
                else:
                    st.metric("Classes with Tutor Preference", "0/0")
            
            with col2:
                st.markdown("**Academic Preferences**")
                assigned_df = results_df[results_df['Assigned Tutor'] != 'UNASSIGNED'].copy()
                
                if len(assigned_df) > 0:
                    # Convert to numeric, treating N/A as NaN
                    assigned_df['Academic Rating Numeric'] = pd.to_numeric(assigned_df['Academic Rating'], errors='coerce')
                    valid_ratings = assigned_df['Academic Rating Numeric'].dropna()
                    
                    if len(valid_ratings) > 0:
                        avg_rating = valid_ratings.mean()
                        st.metric("Average Academic Rating", f"{avg_rating:.2f}/10")
                        
                        # Rating breakdown
                        excellent = sum(valid_ratings >= 9)
                        good = sum((valid_ratings >= 7) & (valid_ratings < 9))
                        acceptable = sum((valid_ratings >= 5) & (valid_ratings < 7))
                        
                        st.write(f"- Excellent (≥9): {excellent}")
                        st.write(f"- Good (7-9): {good}")
                        st.write(f"- Acceptable (5-7): {acceptable}")
            
            # Course diversity
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
                    
                    # Find qualified tutors
                    if course_level == 'PG':
                        qualified_tutors = [
                            t for t, courses_pref in tutor_preferences.items() 
                            if course in courses_pref 
                            and degrees.get(t, '') == 'PhD'
                            and academic_preferences.get((t, course), 5.0) > params.get('theta', 2.0)
                        ]
                    else:
                        qualified_tutors = [
                            t for t, courses_pref in tutor_preferences.items() 
                            if course in courses_pref
                            and academic_preferences.get((t, course), 5.0) > params.get('theta', 2.0)
                        ]
                    
                    if len(qualified_tutors) == 0:
                        if course_level == 'PG':
                            reason = "No PhD tutors with preference passing academic veto"
                        else:
                            reason = "No tutors with preference passing academic veto"
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
                
                st.subheader("📋 Unassigned Classes by Course")
                unassigned_by_course = unassigned_df.groupby(['Course', 'Level']).size().reset_index(name='Unassigned Count')
                st.dataframe(unassigned_by_course, use_container_width=True, hide_index=True)
            else:
                st.success("🎉 All classes successfully assigned!")
            
            # Download results
            st.subheader("📥 Download Results")
            
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='Assignments', index=False)
                workload_df.to_excel(writer, sheet_name='Tutor Workload', index=False)
                if solution['unassigned_classes']:
                    unassigned_df.to_excel(writer, sheet_name='Unassigned', index=False)
                    unassigned_by_course.to_excel(writer, sheet_name='Unassigned by Course', index=False)
                
                # Add parameters sheet
                params_df = pd.DataFrame([params])
                params_df.to_excel(writer, sheet_name='Parameters', index=False)
            
            st.download_button(
                label="📥 Download Results (Excel)",
                data=output.getvalue(),
                file_name="tutor_assignments_T3_dual_preferences.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        else:
            st.error(f"❌ Optimization failed: {solution['status']}")
    
    # Navigation
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to Analysis"):
            # Clear optimization results when going back
            if 'optimization_results' in st.session_state:
                del st.session_state.optimization_results
            st.session_state.step = 5
            st.rerun()
    
    with col2:
        if st.button("Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()
