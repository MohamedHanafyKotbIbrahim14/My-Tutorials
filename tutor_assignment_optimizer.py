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
    """Tutor-Class Assignment using Linear Programming with dual preferences, time conflicts, and degree requirements."""
    
    def __init__(self, classes_df: pd.DataFrame, tutors_df: pd.DataFrame, 
                 preferences: Dict[Tuple[str, str], float],
                 tutor_max_classes: Dict[str, int],
                 tutor_degrees: Dict[str, str],
                 course_diversity_penalty: float = 5.0,
                 phd_priority_bonus: float = 10.0,
                 master_priority_bonus: float = 5.0):
        """
        Initialize the LP problem.
        """
        self.classes_df = classes_df
        self.tutors_df = tutors_df
        self.preferences = preferences
        self.tutor_max_classes = tutor_max_classes
        self.tutor_degrees = tutor_degrees
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
        """Pre-calculate all eligible (tutor, course, class_id) combinations."""
        eligible = []
        
        for tutor in self.tutors:
            for idx, row in self.classes_df.iterrows():
                course = row['course']
                class_id = row['class_id']
                
                # Check preference
                if self.preferences.get((tutor, course), 0) <= 0:
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
        """Build the LP model - optimized version."""
        self.model = pulp.LpProblem("Tutor_Assignment", pulp.LpMaximize)
        
        # Create decision variables only for eligible assignments
        for tutor, course, class_id in self.eligible_assignments:
            var_name = f"x_{len(self.x_vars)}"  # Shorter variable names
            self.x_vars[(tutor, course, class_id)] = pulp.LpVariable(var_name, cat='Binary')
        
        # Create course indicator variables
        tutor_course_pairs = set((t, c) for t, c, _ in self.eligible_assignments)
        for tutor, course in tutor_course_pairs:
            var_name = f"y_{len(self.y_vars)}"
            self.y_vars[(tutor, course)] = pulp.LpVariable(var_name, cat='Binary')
        
        # Objective function - now uses the preference values which include both tutor and academic ratings
        preference_term = pulp.lpSum([
            self.preferences.get((tutor, course), 0) * self.x_vars[(tutor, course, class_id)]
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
        
        diversity_penalty_term = self.course_diversity_penalty * pulp.lpSum([
            self.y_vars[(tutor, course)]
            for (tutor, course) in self.y_vars.keys()
        ])
        
        self.model += preference_term + phd_bonus_term + master_bonus_term - diversity_penalty_term
        
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


def get_star_display(rating: int) -> str:
    """Convert numeric rating to star display."""
    full_stars = rating
    empty_stars = 10 - rating
    return "★" * full_stars + "☆" * empty_stars


def get_rating_emoji(rating: int) -> str:
    """Get emoji based on rating."""
    if rating >= 9:
        return "🌟"
    elif rating >= 7:
        return "✨"
    elif rating >= 5:
        return "⭐"
    elif rating >= 3:
        return "⚠️"
    else:
        return "❌"


def main():
    st.set_page_config(
        page_title="Tutor Assignment Optimizer",
        page_icon="👨‍🏫",
        layout="wide"
    )
    
    st.title("👨‍🏫 Tutor-Class Assignment Optimizer (T3)")
    st.markdown("**Linear Programming with Dual Preferences, Time Conflicts & Degree Requirements**")
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
                st.session_state.preferences = preferences
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
    """Step 4: Set academic preferences for tutors."""
    st.header("Step 4: Academic/Coordinator Preferences for Tutors")
    
    classes_df = st.session_state.classes_df
    tutors_df = st.session_state.tutors_df
    preferences = st.session_state.preferences
    degrees = st.session_state.degrees
    
    st.markdown("""
    **Rate how much you want each tutor to teach each course.**
    
    **Rating Scale (1-10):**
    - 🌟 **10 - Strongly Prefer**: Excellent past performance, first choice
    - ✨ **8-9 - Prefer**: Good choice, strong preference
    - ⭐ **5-7 - Neutral/Acceptable**: Okay, will work
    - ⚠️ **3-4 - Less Preferred**: Concerns, use if needed
    - ❌ **1-2 - Avoid**: Significant concerns, last resort (can be blocked with veto)
    
    **All tutors start at 5 (Neutral) by default.**
    """)
    
    st.info("💡 **Tip**: Higher ratings give more weight in the optimization. The optimizer will try to match high-rated tutors to courses when possible.")
    
    st.markdown("---")
    
    # Initialize academic preferences if not exists
    if 'academic_preferences' not in st.session_state:
        # Default all to 5 (neutral)
        st.session_state.academic_preferences = {}
        for tutor in tutors_df['tutor_name'].unique():
            for course in preferences.get(tutor, []):
                st.session_state.academic_preferences[(course, tutor)] = 5
    
    # Get unique courses
    unique_courses = sorted(classes_df['course'].unique())
    
    # Course selection
    st.subheader("Select Course to Rate")
    selected_course = st.selectbox(
        "Choose a course:",
        options=unique_courses,
        key='selected_course_for_rating'
    )
    
    st.markdown("---")
    
    # Show tutors for selected course
    course_level = classes_df[classes_df['course'] == selected_course].iloc[0]['course_level']
    
    st.subheader(f"Rate Tutors for {selected_course} ({course_level})")
    
    # Filter tutors who can teach this course
    eligible_tutors = []
    for tutor in tutors_df['tutor_name'].unique():
        # Check if tutor has preference for this course
        if selected_course in preferences.get(tutor, []):
            # Check degree eligibility
            tutor_degree = degrees.get(tutor, '')
            if course_level == 'PG' and tutor_degree != 'PhD':
                continue  # Skip non-PhD for PG courses
            eligible_tutors.append(tutor)
    
    if len(eligible_tutors) == 0:
        st.warning(f"⚠️ No eligible tutors found for {selected_course}!")
        if course_level == 'PG':
            st.info("This is a PG course - only PhD tutors who expressed preference can teach it.")
    else:
        st.success(f"✅ Found {len(eligible_tutors)} eligible tutor(s) for this course")
        
        # Bulk actions
        st.markdown("**Quick Actions:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("Set All to 10 (Strongly Prefer)", key=f"set_all_10_{selected_course}"):
                for tutor in eligible_tutors:
                    st.session_state.academic_preferences[(selected_course, tutor)] = 10
                st.rerun()
        
        with col2:
            if st.button("Set All to 7 (Prefer)", key=f"set_all_7_{selected_course}"):
                for tutor in eligible_tutors:
                    st.session_state.academic_preferences[(selected_course, tutor)] = 7
                st.rerun()
        
        with col3:
            if st.button("Set All to 5 (Neutral)", key=f"set_all_5_{selected_course}"):
                for tutor in eligible_tutors:
                    st.session_state.academic_preferences[(selected_course, tutor)] = 5
                st.rerun()
        
        with col4:
            if st.button("Set All to 3 (Less Prefer)", key=f"set_all_3_{selected_course}"):
                for tutor in eligible_tutors:
                    st.session_state.academic_preferences[(selected_course, tutor)] = 3
                st.rerun()
        
        st.markdown("---")
        
        # Individual tutor ratings
        st.markdown("**Individual Ratings:**")
        
        # Sort tutors by current rating (highest first)
        eligible_tutors_sorted = sorted(
            eligible_tutors,
            key=lambda t: st.session_state.academic_preferences.get((selected_course, t), 5),
            reverse=True
        )
        
        for tutor in eligible_tutors_sorted:
            current_rating = st.session_state.academic_preferences.get((selected_course, tutor), 5)
            tutor_degree = degrees.get(tutor, 'Not Specified')
            
            col1, col2, col3, col4 = st.columns([3, 2, 3, 2])
            
            with col1:
                st.markdown(f"**{tutor}**")
                st.caption(f"Degree: {tutor_degree}")
            
            with col2:
                rating_emoji = get_rating_emoji(current_rating)
                stars = get_star_display(current_rating)
                st.markdown(f"{rating_emoji} **{current_rating}/10**")
                st.caption(stars)
            
            with col3:
                new_rating = st.slider(
                    "Rating",
                    min_value=1,
                    max_value=10,
                    value=current_rating,
                    key=f"rating_{selected_course}_{tutor}",
                    label_visibility="collapsed"
                )
                
                if new_rating != current_rating:
                    st.session_state.academic_preferences[(selected_course, tutor)] = new_rating
            
            with col4:
                if new_rating >= 8:
                    st.success("Prefer")
                elif new_rating >= 5:
                    st.info("Neutral")
                else:
                    st.warning("Less Prefer")
            
            st.markdown("---")
    
    # Summary statistics
    st.subheader("📊 Overall Summary")
    
    # Count ratings across all courses
    rating_counts = {i: 0 for i in range(1, 11)}
    for rating in st.session_state.academic_preferences.values():
        rating_counts[rating] = rating_counts.get(rating, 0) + 1
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        high_ratings = sum(v for k, v in rating_counts.items() if k >= 8)
        st.metric("Preferred (8-10)", high_ratings)
        st.caption("🌟 Strong preference")
    
    with col2:
        neutral_ratings = sum(v for k, v in rating_counts.items() if 5 <= k <= 7)
        st.metric("Neutral (5-7)", neutral_ratings)
        st.caption("⭐ Acceptable")
    
    with col3:
        low_ratings = sum(v for k, v in rating_counts.items() if k < 5)
        st.metric("Less Preferred (1-4)", low_ratings)
        st.caption("⚠️ Use if needed")
    
    with col4:
        total_ratings = len(st.session_state.academic_preferences)
        st.metric("Total Ratings", total_ratings)
        st.caption("📝 All tutor-course pairs")
    
    # Show rating distribution chart
    with st.expander("📈 View Rating Distribution"):
        rating_dist_data = pd.DataFrame([
            {'Rating': k, 'Count': v} 
            for k, v in rating_counts.items()
        ])
        
        fig = px.bar(
            rating_dist_data,
            x='Rating',
            y='Count',
            title='Distribution of Academic Ratings',
            labels={'Rating': 'Rating (1-10)', 'Count': 'Number of Tutor-Course Pairs'},
            color='Rating',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Show all ratings table
    with st.expander("📋 View All Ratings"):
        all_ratings_data = []
        for (course, tutor), rating in sorted(st.session_state.academic_preferences.items()):
            tutor_degree = degrees.get(tutor, 'N/A')
            all_ratings_data.append({
                'Course': course,
                'Tutor': tutor,
                'Degree': tutor_degree,
                'Rating': rating,
                'Stars': get_star_display(rating),
                'Status': get_rating_emoji(rating)
            })
        
        all_ratings_df = pd.DataFrame(all_ratings_data)
        st.dataframe(all_ratings_df, use_container_width=True, hide_index=True)
    
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
    preferences = st.session_state.preferences
    max_classes = st.session_state.max_classes
    degrees = st.session_state.degrees
    academic_preferences = st.session_state.get('academic_preferences', {})
    
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
    
    if total_capacity < len(classes_df):
        st.warning(f"⚠️ Total tutor capacity ({total_capacity}) is less than total classes ({len(classes_df)}). Optimization will proceed and show unassigned classes.")
    else:
        surplus = total_capacity - len(classes_df)
        st.success(f"✅ Sufficient capacity: {surplus} extra class slots available")
    
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
    
    st.subheader("📋 Course Coverage Analysis (with Degree Requirements & Academic Preferences)")
    
    st.info("ℹ️ Even if some courses show insufficient capacity, optimization will still run and show which classes remain unassigned.")
    
    coverage_data = []
    courses = sorted(classes_df['course'].unique())
    
    for course in courses:
        course_df = classes_df[classes_df['course'] == course]
        num_classes = len(course_df)
        course_level = course_df.iloc[0]['course_level']
        
        if course_level == 'PG':
            qualified_tutors = [
                tutor for tutor in tutors_df['tutor_name'].unique()
                if course in preferences.get(tutor, []) and degrees.get(tutor, '') == 'PhD'
            ]
        else:
            qualified_tutors = [
                tutor for tutor in tutors_df['tutor_name'].unique()
                if course in preferences.get(tutor, [])
            ]
        
        num_qualified = len(qualified_tutors)
        total_capacity_for_course = sum([max_classes.get(tutor, 0) for tutor in qualified_tutors])
        
        # Calculate average academic rating for this course
        course_ratings = [
            academic_preferences.get((course, tutor), 5)
            for tutor in qualified_tutors
        ]
        avg_rating = sum(course_ratings) / len(course_ratings) if course_ratings else 5
        
        coverage_data.append({
            'Course': course,
            'Level': course_level,
            'Classes': num_classes,
            'Qualified Tutors': num_qualified,
            'Tutor Capacity': total_capacity_for_course,
            'Avg Academic Rating': f"{avg_rating:.1f}",
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
                st.error(f"**{row['Course']}**: No PhD tutors available who prefer this course!")
            elif row['Qualified Tutors'] == 0:
                st.error(f"**{row['Course']}**: No tutors have expressed preference for this course!")
        
        st.info("💡 The optimization will still run and show which specific classes cannot be assigned.")
    
    # Academic Preferences Summary
    st.subheader("🎓 Academic Preferences Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Show top-rated tutors
        tutor_avg_ratings = {}
        for (course, tutor), rating in academic_preferences.items():
            if tutor not in tutor_avg_ratings:
                tutor_avg_ratings[tutor] = []
            tutor_avg_ratings[tutor].append(rating)
        
        tutor_avg_ratings_computed = {
            tutor: sum(ratings) / len(ratings)
            for tutor, ratings in tutor_avg_ratings.items()
        }
        
        top_tutors = sorted(tutor_avg_ratings_computed.items(), key=lambda x: x[1], reverse=True)[:10]
        
        st.markdown("**Top 10 Highest-Rated Tutors (Average):**")
        for tutor, avg_rating in top_tutors:
            stars = get_star_display(int(round(avg_rating)))
            st.write(f"{get_rating_emoji(int(round(avg_rating)))} **{tutor}**: {avg_rating:.1f}/10 {stars}")
    
    with col2:
        # Show courses with highest average ratings
        course_avg_ratings = {}
        for (course, tutor), rating in academic_preferences.items():
            if course not in course_avg_ratings:
                course_avg_ratings[course] = []
            course_avg_ratings[course].append(rating)
        
        course_avg_ratings_computed = {
            course: sum(ratings) / len(ratings)
            for course, ratings in course_avg_ratings.items()
        }
        
        top_courses = sorted(course_avg_ratings_computed.items(), key=lambda x: x[1], reverse=True)
        
        st.markdown("**Courses by Average Academic Rating:**")
        for course, avg_rating in top_courses:
            stars = get_star_display(int(round(avg_rating)))
            st.write(f"{get_rating_emoji(int(round(avg_rating)))} **{course}**: {avg_rating:.1f}/10")
    
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
    preferences = st.session_state.preferences
    max_classes = st.session_state.max_classes
    degrees = st.session_state.degrees
    academic_preferences = st.session_state.get('academic_preferences', {})
    
    # Use a form to prevent reruns on input changes
    with st.form("optimization_form"):
        st.subheader("⚙️ Optimization Parameters")
        
        st.markdown("""
        **Dual Preference System:**
        - **Tutor Preference (w₁)**: Weight for tutor's interest in teaching the course
        - **Academic Preference (w₂)**: Weight for coordinator's rating of tutor-course match
        - **Combined Score** = (w₁ × tutor preference) + (w₂ × academic rating) + degree bonuses
        """)
        
        st.markdown("---")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            tutor_preference_weight = st.slider(
                "Tutor Preference Weight (w₁)",
                min_value=0.0,
                max_value=30.0,
                value=10.0,
                step=1.0,
                help="Weight for tutor's interest in teaching (binary yes/no = 10 points if yes)"
            )
            
            st.info(f"""
            **Current: {tutor_preference_weight}**
            
            - **0**: Ignore tutors
            - **10**: Standard
            - **20**: High value
            """)
        
        with col2:
            academic_preference_weight = st.slider(
                "Academic Preference Weight (w₂)",
                min_value=0.0,
                max_value=30.0,
                value=20.0,
                step=1.0,
                help="Weight for academic/coordinator ratings (1-10 scale)"
            )
            
            st.success(f"""
            **Current: {academic_preference_weight}**
            
            - **0**: Ignore academic
            - **10**: Equal to tutor
            - **20**: 2× tutor (recommended)
            """)
        
        with col3:
            phd_priority_bonus = st.slider(
                "PhD Priority Bonus (β)",
                min_value=0.0,
                max_value=20.0,
                value=20.0,
                step=1.0,
                help="Bonus points for assigning PhD students"
            )
            
            st.success(f"""
            **Current: {phd_priority_bonus}**
            
            - **0**: No priority
            - **10**: Strong (recommended)
            - **20**: Very strong
            """)
        
        with col4:
            master_priority_bonus = st.slider(
                "Master Priority Bonus (γ)",
                min_value=0.0,
                max_value=15.0,
                value=15.0,
                step=1.0,
                help="Bonus points for Master students (should be < PhD bonus)"
            )
            
            st.info(f"""
            **Current: {master_priority_bonus}**
            
            - **0**: No priority
            - **5**: Moderate (recommended)
            - **10**: Strong
            """)
        
        with col5:
            course_diversity_penalty = st.slider(
                "Diversity Penalty (α)",
                min_value=0.0,
                max_value=20.0,
                value=5.0,
                step=0.5,
                help="Penalty for teaching multiple different courses"
            )
            
            st.warning(f"""
            **Current: {course_diversity_penalty}**
            
            - **0**: No penalty
            - **5**: Moderate
            - **10+**: Strong
            """)
        
        st.markdown("---")
        
        # Academic veto threshold
        st.subheader("🚫 Academic Veto Threshold")
        
        col1, col2 = st.columns([2, 3])
        
        with col1:
            veto_threshold = st.slider(
                "Veto Threshold (θ)",
                min_value=0,
                max_value=4,
                value=2,
                step=1,
                help="Block tutors with academic rating ≤ threshold (0 = disabled)"
            )
        
        with col2:
            if veto_threshold == 0:
                st.info("✅ **Veto disabled** - All tutors are eligible regardless of rating")
            elif veto_threshold == 1:
                st.warning(f"⚠️ **Blocking tutors rated 1** - Only tutors with severe concerns are blocked")
            elif veto_threshold == 2:
                st.warning(f"⚠️ **Blocking tutors rated 1-2** - Tutors with significant concerns are blocked (recommended)")
            elif veto_threshold == 3:
                st.error(f"❌ **Blocking tutors rated 1-3** - Moderately restrictive")
            else:
                st.error(f"❌ **Blocking tutors rated 1-4** - Very restrictive, only accepts neutral+ ratings")
        
        st.markdown("---")
        
        # Example scoring
        st.subheader("📊 Example Scoring with Current Weights")
        
        example_col1, example_col2, example_col3 = st.columns(3)
        
        with example_col1:
            st.markdown("**Best Case (PhD, perfect match):**")
            best_score = (tutor_preference_weight * 10) + (academic_preference_weight * 10) + phd_priority_bonus
            st.metric("Total Score", f"{best_score:.0f} points")
            st.caption("Tutor: Yes (10) + Academic: 10 + PhD bonus")
        
        with example_col2:
            st.markdown("**Good Case (Master, strong match):**")
            good_score = (tutor_preference_weight * 10) + (academic_preference_weight * 8) + master_priority_bonus
            st.metric("Total Score", f"{good_score:.0f} points")
            st.caption("Tutor: Yes (10) + Academic: 8 + Master bonus")
        
        with example_col3:
            st.markdown("**Acceptable Case (Bachelor, neutral):**")
            acceptable_score = (tutor_preference_weight * 10) + (academic_preference_weight * 5) + 0
            st.metric("Total Score", f"{acceptable_score:.0f} points")
            st.caption("Tutor: Yes (10) + Academic: 5 + No bonus")
        
        # Submit button
        run_optimization = st.form_submit_button("🚀 Run Optimization", type="primary")
    
    # Only run optimization when button is clicked
    if run_optimization or 'optimization_results' in st.session_state:
        
        if run_optimization:
            # Clear previous results
            if 'optimization_results' in st.session_state:
                del st.session_state.optimization_results
            
            st.markdown("---")
            
            # Build preference dictionary with dual preferences and veto
            pref_dict = {}
            vetoed_count = 0
            
            for tutor, courses in preferences.items():
                for course in courses:
                    # Get academic rating (default 5 if not set)
                    academic_rating = academic_preferences.get((course, tutor), 5)
                    
                    # Check veto threshold
                    if academic_rating <= veto_threshold:
                        vetoed_count += 1
                        continue  # Skip this tutor-course pair (vetoed)
                    
                    # Tutor preference: 10 if they expressed interest (binary)
                    tutor_pref = 10
                    
                    # Combined score: (w1 × p_tc) + (w2 × a_tc)
                    pref_dict[(tutor, course)] = (tutor_preference_weight * tutor_pref) + (academic_preference_weight * academic_rating)
            
            if vetoed_count > 0:
                st.warning(f"🚫 Academic veto applied: {vetoed_count} tutor-course pair(s) blocked (rating ≤ {veto_threshold})")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Initializing optimization model...")
                progress_bar.progress(20)
                
                lp = TutorAssignmentLP(
                    classes_df=classes_df,
                    tutors_df=tutors_df,
                    preferences=pref_dict,
                    tutor_max_classes=max_classes,
                    tutor_degrees=degrees,
                    course_diversity_penalty=course_diversity_penalty,
                    phd_priority_bonus=phd_priority_bonus,
                    master_priority_bonus=master_priority_bonus
                )
                
                status_text.text("Building model...")
                progress_bar.progress(40)
                lp.build_model()
                
                num_vars = len(lp.x_vars) + len(lp.y_vars)
                st.info(f"📊 Model created with {num_vars} decision variables ({len(lp.x_vars)} assignments + {len(lp.y_vars)} course indicators)")
                
                status_text.text("Solving optimization problem... (no time limit - finding optimal solution)")
                progress_bar.progress(60)
                
                # Call solve without time limit (will run until optimal)
                solution = lp.solve(time_limit=None)
                
                progress_bar.progress(100)
                status_text.text(f"✅ Optimization completed in {solution.get('solve_time', 0):.1f} seconds")
                
                # Store results and parameters in session state
                st.session_state.optimization_results = solution
                st.session_state.optimization_params = {
                    'tutor_weight': tutor_preference_weight,
                    'academic_weight': academic_preference_weight,
                    'phd_bonus': phd_priority_bonus,
                    'master_bonus': master_priority_bonus,
                    'diversity_penalty': course_diversity_penalty,
                    'veto_threshold': veto_threshold,
                    'vetoed_count': vetoed_count
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
        
        # Results summary
        st.subheader("📈 Optimization Summary")
        
        col1, col2, col3, col4, col5 = st.columns(5)
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
        with col5:
            if params.get('vetoed_count', 0) > 0:
                st.metric("Vetoed Assignments", params['vetoed_count'])
                st.caption("🚫 Blocked by threshold")
        
        if solution['status'] in ['Optimal', 'Not Solved']:
            if unassigned_count == 0:
                st.success("✅ All classes assigned!")
            else:
                st.warning(f"⚠️ {unassigned_count} classes could not be assigned (see details below).")
            
            # Parameter summary
            with st.expander("⚙️ View Optimization Parameters Used"):
                param_col1, param_col2 = st.columns(2)
                with param_col1:
                    st.markdown("**Preference Weights:**")
                    st.write(f"- Tutor Preference Weight (w₁): {params.get('tutor_weight', 10)}")
                    st.write(f"- Academic Preference Weight (w₂): {params.get('academic_weight', 20)}")
                    st.markdown("**Degree Priority Bonuses:**")
                    st.write(f"- PhD Priority Bonus (β): {params.get('phd_bonus', 10)}")
                    st.write(f"- Master Priority Bonus (γ): {params.get('master_bonus', 5)}")
                with param_col2:
                    st.markdown("**Other Parameters:**")
                    st.write(f"- Course Diversity Penalty (α): {params.get('diversity_penalty', 5)}")
                    st.write(f"- Academic Veto Threshold (θ): {params.get('veto_threshold', 2)}")
                    if params.get('vetoed_count', 0) > 0:
                        st.write(f"- Assignments Vetoed: {params['vetoed_count']}")
            
            # Create results dataframe
            results_data = []
            for idx, row in classes_df.iterrows():
                course = row['course']
                class_id = row['class_id']
                assigned_tutor = solution['assignments'].get((course, class_id), 'UNASSIGNED')
                
                tutor_degree = degrees.get(assigned_tutor, 'N/A') if assigned_tutor != 'UNASSIGNED' else 'N/A'
                
                # Get academic rating and tutor preference for this assignment
                if assigned_tutor != 'UNASSIGNED':
                    academic_rating = academic_preferences.get((course, assigned_tutor), 5)
                    tutor_pref = 10  # Binary: has preference
                    rating_display = f"{academic_rating} {get_star_display(academic_rating)}"
                    combined_score = (params.get('tutor_weight', 10) * tutor_pref) + (params.get('academic_weight', 20) * academic_rating)
                    score_display = f"{combined_score:.0f}"
                else:
                    rating_display = 'N/A'
                    score_display = 'N/A'
                
                results_data.append({
                    'Course': course,
                    'Level': row['course_level'],
                    'Class ID': class_id,
                    'Type': row['type'],
                    'Section': row['section'],
                    'Time': row['time'],
                    'Assigned Tutor': assigned_tutor,
                    'Tutor Degree': tutor_degree,
                    'Academic Rating': rating_display,
                    'Combined Score': score_display
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
            
            # Academic Rating & Combined Score Satisfaction Analysis
            st.subheader("🎓 Preference Satisfaction Analysis")
            
            assigned_df = results_df[results_df['Assigned Tutor'] != 'UNASSIGNED'].copy()
            
            # Parse ratings from display string
            assigned_df['Rating_Numeric'] = assigned_df['Academic Rating'].apply(
                lambda x: int(x.split()[0]) if x != 'N/A' else 0
            )
            
            assigned_df['Score_Numeric'] = assigned_df['Combined Score'].apply(
                lambda x: float(x) if x != 'N/A' else 0
            )
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                high_satisfaction = len(assigned_df[assigned_df['Rating_Numeric'] >= 8])
                st.metric("High Academic Rating (8-10)", high_satisfaction)
                st.caption(f"{high_satisfaction/len(assigned_df)*100:.1f}% of assignments" if len(assigned_df) > 0 else "N/A")
            
            with col2:
                medium_satisfaction = len(assigned_df[(assigned_df['Rating_Numeric'] >= 5) & (assigned_df['Rating_Numeric'] < 8)])
                st.metric("Medium Academic Rating (5-7)", medium_satisfaction)
                st.caption(f"{medium_satisfaction/len(assigned_df)*100:.1f}% of assignments" if len(assigned_df) > 0 else "N/A")
            
            with col3:
                low_satisfaction = len(assigned_df[assigned_df['Rating_Numeric'] < 5])
                st.metric("Low Academic Rating (1-4)", low_satisfaction)
                st.caption(f"{low_satisfaction/len(assigned_df)*100:.1f}% of assignments" if len(assigned_df) > 0 else "N/A")
            
            with col4:
                avg_rating = assigned_df['Rating_Numeric'].mean() if len(assigned_df) > 0 else 0
                st.metric("Avg Academic Rating", f"{avg_rating:.1f}/10")
                st.caption(get_star_display(int(round(avg_rating))))
            
            # Combined score distribution
            if len(assigned_df) > 0:
                st.markdown("**Combined Score Distribution:**")
                avg_combined_score = assigned_df['Score_Numeric'].mean()
                max_combined_score = assigned_df['Score_Numeric'].max()
                min_combined_score = assigned_df['Score_Numeric'].min()
                
                score_col1, score_col2, score_col3 = st.columns(3)
                with score_col1:
                    st.metric("Average Combined Score", f"{avg_combined_score:.0f}")
                with score_col2:
                    st.metric("Maximum Score", f"{max_combined_score:.0f}")
                with score_col3:
                    st.metric("Minimum Score", f"{min_combined_score:.0f}")
            
            # Tutor workload summary
            st.subheader("👥 Tutor Workload Summary")
            
            workload_data = []
            for tutor in tutors_df['tutor_name'].unique():
                load = solution['tutor_loads'][tutor]
                assigned_classes = load['classes']
                
                courses_count = {}
                pg_count = 0
                ug_count = 0
                ratings_for_tutor = []
                scores_for_tutor = []
                
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
                    
                    # Get academic rating and combined score
                    rating = academic_preferences.get((course, tutor), 5)
                    ratings_for_tutor.append(rating)
                    
                    combined = (params.get('tutor_weight', 10) * 10) + (params.get('academic_weight', 20) * rating)
                    scores_for_tutor.append(combined)
                
                courses_str = ', '.join([f"{course}({count})" for course, count in courses_count.items()])
                num_different_courses = solution['tutor_course_diversity'].get(tutor, 0)
                
                tutor_degree = degrees.get(tutor, 'Not Specified')
                
                avg_rating = sum(ratings_for_tutor) / len(ratings_for_tutor) if ratings_for_tutor else 0
                avg_score = sum(scores_for_tutor) / len(scores_for_tutor) if scores_for_tutor else 0
                
                workload_data.append({
                    'Tutor': tutor,
                    'Degree': tutor_degree,
                    'Different Courses': num_different_courses,
                    'PG Classes': pg_count,
                    'UG Classes': ug_count,
                    'Total Classes': load['total'],
                    'Max Allowed': max_classes.get(tutor, 0),
                    'Utilization': f"{load['total']}/{max_classes.get(tutor, 0)}",
                    'Avg Academic Rating': f"{avg_rating:.1f}" if avg_rating > 0 else "N/A",
                    'Avg Combined Score': f"{avg_score:.0f}" if avg_score > 0 else "N/A",
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
                    
                    if course_level == 'PG':
                        qualified_tutors = [
                            t for t, courses_pref in preferences.items() 
                            if course in courses_pref and degrees.get(t, '') == 'PhD'
                        ]
                    else:
                        qualified_tutors = [t for t, courses_pref in preferences.items() if course in courses_pref]
                    
                    # Check if any were vetoed
                    vetoed_for_course = []
                    for tutor in tutors_df['tutor_name'].unique():
                        if course in preferences.get(tutor, []):
                            if degrees.get(tutor, '') != 'PhD' and course_level == 'PG':
                                continue
                            rating = academic_preferences.get((course, tutor), 5)
                            if rating <= params.get('veto_threshold', 2):
                                vetoed_for_course.append(tutor)
                    
                    if len(qualified_tutors) == 0:
                        if course_level == 'PG':
                            reason = "No PhD tutors with preference for this course"
                        else:
                            reason = "No tutors with preference for this course"
                    elif len(vetoed_for_course) > 0:
                        reason = f"Academic veto blocked {len(vetoed_for_course)} tutor(s)"
                    else:
                        reason = "Time conflict or capacity exceeded"
                    
                    unassigned_data.append({
                        'Course': course,
                        'Level': course_level,
                        'Class ID': class_id,
                        'Section': class_row['section'],
                        'Time': class_row['time'],
                        'Qualified Tutors': len(qualified_tutors),
                        'Vetoed Tutors': len(vetoed_for_course),
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
                
                # Add academic preferences sheet
                academic_prefs_export = []
                for (course, tutor), rating in sorted(academic_preferences.items()):
                    academic_prefs_export.append({
                        'Course': course,
                        'Tutor': tutor,
                        'Academic Rating': rating
                    })
                academic_prefs_df = pd.DataFrame(academic_prefs_export)
                academic_prefs_df.to_excel(writer, sheet_name='Academic Preferences', index=False)
                
                # Add optimization parameters sheet
                params_export = pd.DataFrame([{
                    'Parameter': 'Tutor Preference Weight (w1)',
                    'Value': params.get('tutor_weight', 10),
                    'Description': 'Weight for tutor interest (binary yes/no)'
                }, {
                    'Parameter': 'Academic Preference Weight (w2)',
                    'Value': params.get('academic_weight', 20),
                    'Description': 'Weight for academic ratings (1-10 scale)'
                }, {
                    'Parameter': 'PhD Priority Bonus (β)',
                    'Value': params.get('phd_bonus', 10),
                    'Description': 'Bonus points for PhD students'
                }, {
                    'Parameter': 'Master Priority Bonus (γ)',
                    'Value': params.get('master_bonus', 5),
                    'Description': 'Bonus points for Master students'
                }, {
                    'Parameter': 'Course Diversity Penalty (α)',
                    'Value': params.get('diversity_penalty', 5),
                    'Description': 'Penalty for teaching multiple courses'
                }, {
                    'Parameter': 'Academic Veto Threshold (θ)',
                    'Value': params.get('veto_threshold', 2),
                    'Description': 'Block tutors with rating ≤ threshold'
                }, {
                    'Parameter': 'Vetoed Assignments',
                    'Value': params.get('vetoed_count', 0),
                    'Description': 'Number of tutor-course pairs blocked'
                }])
                params_export.to_excel(writer, sheet_name='Optimization Parameters', index=False)
            
            st.download_button(
                label="📥 Download Complete Results (Excel)",
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
            if 'optimization_params' in st.session_state:
                del st.session_state.optimization_params
            st.session_state.step = 5
            st.rerun()
    
    with col2:
        if st.button("Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()
