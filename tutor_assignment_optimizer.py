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
    """Tutor-Class Assignment using Linear Programming with time conflict detection and degree requirements."""
    
    def __init__(self, classes_df: pd.DataFrame, tutors_df: pd.DataFrame, 
                 preferences: Dict[Tuple[str, str], float],
                 tutor_max_classes: Dict[str, int],
                 tutor_degrees: Dict[str, str],
                 course_diversity_penalty: float = 5.0,
                 phd_priority_bonus: float = 2.0):
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
        
        # Objective function
        preference_term = pulp.lpSum([
            10 * self.x_vars[(tutor, course, class_id)]
            for (tutor, course, class_id) in self.x_vars.keys()
        ])
        
        phd_bonus_term = self.phd_priority_bonus * pulp.lpSum([
            self.x_vars[(tutor, course, class_id)]
            for (tutor, course, class_id) in self.x_vars.keys()
            if self.tutor_degrees.get(tutor, '') == 'PhD'
        ])
        
        diversity_penalty_term = self.course_diversity_penalty * pulp.lpSum([
            self.y_vars[(tutor, course)]
            for (tutor, course) in self.y_vars.keys()
        ])
        
        self.model += preference_term + phd_bonus_term - diversity_penalty_term
        
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
    
    def solve(self, time_limit=180):
        """Solve the optimization problem."""
        if self.model is None:
            self.build_model()
        
        solver = pulp.PULP_CBC_CMD(
            msg=0,  # Suppress output
            timeLimit=time_limit,
            gapRel=0.02,  # Stop if within 2% of optimal
            threads=4  # Use multiple threads
        )
        
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
    """Normalize degree text to standard categories."""
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
    
    if 'step' not in st.session_state:
        st.session_state.step = 1
    
    if st.session_state.step == 1:
        show_upload_step()
    elif st.session_state.step == 2:
        show_course_level_step()
    elif st.session_state.step == 3:
        show_review_tutors_step()
    elif st.session_state.step == 4:
        show_analysis_step()
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
    """)
    
    st.markdown("---")
    
    needs_review_max = tutors_df[tutors_df['max_classes_status'].str.contains('defaulted|Could not parse', case=False, na=False)]
    
    if len(needs_review_max) > 0:
        st.warning(f"⚠️ {len(needs_review_max)} tutor(s) have max_classes values that may need correction")
    else:
        st.success("✅ All max_classes values parsed successfully!")
    
    st.markdown("---")
    
    st.subheader("Edit Tutor Information")
    
    edited_degrees = {}
    edited_max_classes = {}
    
    for idx, row in tutors_df.iterrows():
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 3, 1])
            
            with col1:
                st.markdown(f"**{row['tutor_name']}**")
                st.caption(f"Preferences: {row['t3_courses'][:50]}...")
            
            with col2:
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
                
                if new_degree == 'PhD':
                    st.caption("✅ Can teach PG & UG")
                else:
                    st.caption("⚠️ Can teach UG only")
                
                edited_degrees[row['tutor_name']] = new_degree
            
            with col3:
                original_text = row['degree'][:30] + "..." if len(row['degree']) > 30 else row['degree']
                st.caption(f"Original: {original_text}")
            
            with col4:
                status = row['max_classes_status']
                
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
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Back to Course Levels"):
            st.session_state.step = 2
            st.rerun()
    
    with col3:
        if st.button("Next: Analysis →", type="primary"):
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
    
    st.subheader("📋 Course Coverage Analysis (with Degree Requirements)")
    
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
                st.error(f"**{row['Course']}**: No PhD tutors available who prefer this course!")
            elif row['Qualified Tutors'] == 0:
                st.error(f"**{row['Course']}**: No tutors have expressed preference for this course!")
        
        st.info("💡 The optimization will still run and show which specific classes cannot be assigned.")
    
    st.markdown("---")
    
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
    """Step 5: Run optimization and show results - WITH FORM TO PREVENT RERUNS."""
    st.header("Step 5: Optimization Results")
    
    classes_df = st.session_state.classes_df
    tutors_df = st.session_state.tutors_df
    preferences = st.session_state.preferences
    max_classes = st.session_state.max_classes
    degrees = st.session_state.degrees
    
    # Use a form to prevent reruns on input changes
    with st.form("optimization_form"):
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
            
            st.info(f"""
            **Current Setting: {course_diversity_penalty}**
            
            - **0**: No penalty (tutors can teach many courses)
            - **5**: Moderate penalty (recommended)
            - **10+**: Strong penalty (tutors prefer 1-2 courses only)
            """)
        
        with col2:
            phd_priority_bonus = st.slider(
                "PhD Priority Bonus",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.5,
                help="Higher bonus gives more priority to assigning PhD students"
            )
            
            st.success(f"""
            **Current Setting: {phd_priority_bonus}**
            
            - **0**: No priority for PhD students
            - **2**: Moderate priority (recommended)
            - **5+**: Strong priority for PhD students
            
            🎓 PhD students will be assigned first!
            """)
        
        time_limit = st.slider(
            "Optimization Time Limit (seconds)",
            min_value=30,
            max_value=300,
            value=120,
            step=30,
            help="Maximum time for optimization"
        )
        
        # Submit button
        run_optimization = st.form_submit_button("🚀 Run Optimization", type="primary")
    
    # Only run optimization when button is clicked
    if run_optimization or 'optimization_results' in st.session_state:
        
        if run_optimization:
            # Clear previous results
            if 'optimization_results' in st.session_state:
                del st.session_state.optimization_results
            
            st.markdown("---")
            
            # Build preference dictionary
            pref_dict = {}
            for tutor, courses in preferences.items():
                for course in courses:
                    pref_dict[(tutor, course)] = 10
            
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
                    phd_priority_bonus=phd_priority_bonus
                )
                
                status_text.text("Building model...")
                progress_bar.progress(40)
                lp.build_model()
                
                num_vars = len(lp.x_vars) + len(lp.y_vars)
                st.info(f"📊 Model created with {num_vars} decision variables ({len(lp.x_vars)} assignments + {len(lp.y_vars)} course indicators)")
                
                status_text.text("Solving optimization problem...")
                progress_bar.progress(60)
                
                solution = lp.solve(time_limit=time_limit)
                
                progress_bar.progress(100)
                status_text.text(f"✅ Optimization completed in {solution.get('solve_time', 0):.1f} seconds")
                
                # Store results in session state
                st.session_state.optimization_results = solution
                
            except Exception as e:
                progress_bar.progress(100)
                status_text.text("❌ Error occurred")
                st.error(f"❌ Error during optimization: {str(e)}")
                st.exception(e)
                return
        
        # Display results from session state
        solution = st.session_state.optimization_results
        
        st.markdown("---")
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
            
            # PhD Priority Analysis
            st.subheader("🎓 PhD Priority Analysis")
            phd_tutors = workload_df[workload_df['Degree'] == 'PhD']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                phd_assigned = phd_tutors[phd_tutors['Total Classes'] > 0].shape[0]
                phd_total = len(phd_tutors)
                st.metric("PhD Tutors Assigned", f"{phd_assigned}/{phd_total}")
                if phd_total > 0:
                    st.caption(f"{phd_assigned/phd_total*100:.0f}% utilization")
            
            with col2:
                phd_classes = phd_tutors['Total Classes'].sum()
                total_assigned = workload_df['Total Classes'].sum()
                st.metric("Classes Taught by PhD", phd_classes)
                if total_assigned > 0:
                    st.caption(f"{phd_classes/total_assigned*100:.0f}% of all classes")
            
            with col3:
                avg_phd_load = phd_tutors[phd_tutors['Total Classes'] > 0]['Total Classes'].mean() if phd_assigned > 0 else 0
                st.metric("Avg PhD Workload", f"{avg_phd_load:.1f}")
                st.caption("Classes per assigned PhD tutor")
            
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
            
            st.download_button(
                label="📥 Download Results (Excel)",
                data=output.getvalue(),
                file_name="tutor_assignments_T3_optimized.xlsx",
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
            st.session_state.step = 4
            st.rerun()
    
    with col2:
        if st.button("Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()
