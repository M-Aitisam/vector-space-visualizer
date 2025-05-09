import streamlit as st
import numpy as np

st.set_page_config(page_title="Vector Spaces & Subspaces", layout="centered")

st.title("🧮 Vector Spaces and Subspaces Demonstrator")

st.markdown("""
This app helps you understand the fundamental concepts of **vector spaces** and **subspaces** in linear algebra.

You can enter vectors, check for vector space properties, and test span membership.
""")

st.header("1️⃣ Input Vectors in ℝⁿ")

num_vectors = st.number_input("Number of vectors:", min_value=2, max_value=5, value=3)
dimension = st.number_input("Dimension of each vector (n):", min_value=2, max_value=5, value=3)

vectors = []
st.subheader("Enter each vector:")
for i in range(num_vectors):
    v_input = st.text_input(f"Vector {i+1} (comma-separated)", value="1,2,3")
    v = np.array([float(x.strip()) for x in v_input.split(",")])
    if len(v) != dimension:
        st.error(f"Vector {i+1} must have {dimension} elements.")
    else:
        vectors.append(v)

# Scalars for testing
st.header("2️⃣ Scalars for Scalar Multiplication Test")
scalar_input = st.text_input("Enter scalars (comma-separated)", value="1, -1, 2, 0.5")
scalars = [float(s.strip()) for s in scalar_input.split(",")]

def is_closed_under_addition(vectors):
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            sum_vector = vectors[i] + vectors[j]
            if not any(np.allclose(sum_vector, v) for v in vectors):
                return False
    return True

def is_closed_under_scalar_multiplication(vectors, scalars):
    for v in vectors:
        for s in scalars:
            product = s * v
            if not any(np.allclose(product, vec) for vec in vectors):
                return False
    return True

def has_zero_vector(vectors):
    zero = np.zeros_like(vectors[0])
    return any(np.allclose(zero, v) for v in vectors)

def is_in_span(vector, basis_vectors):
    A = np.column_stack(basis_vectors)
    try:
        x = np.linalg.lstsq(A, vector, rcond=None)[0]
        return np.allclose(A @ x, vector), x
    except:
        return False, None

# Analysis Button
if len(vectors) == num_vectors:
    st.header("3️⃣ Vector Space Property Checks")

    col1, col2 = st.columns(2)

    with col1:
        if is_closed_under_addition(vectors):
            st.success("✅ Closure under addition holds.")
        else:
            st.error("❌ Closure under addition fails.")

    with col2:
        if is_closed_under_scalar_multiplication(vectors, scalars):
            st.success("✅ Closure under scalar multiplication holds.")
        else:
            st.error("❌ Closure under scalar multiplication fails.")

    if has_zero_vector(vectors):
        st.success("✅ Zero vector is included.")
    else:
        st.error("❌ Zero vector is not included.")

    if (
        is_closed_under_addition(vectors)
        and is_closed_under_scalar_multiplication(vectors, scalars)
        and has_zero_vector(vectors)
    ):
        st.success("✅ The set **is a subspace** of ℝⁿ.")
    else:
        st.warning("❌ The set **is not a subspace** of ℝⁿ.")

    # Span Checker
    st.header("4️⃣ Test if a Vector is in the Span")

    test_vector_input = st.text_input("Enter test vector (comma-separated)", value="3,6,9")
    test_vector = np.array([float(x.strip()) for x in test_vector_input.split(",")])

    if len(test_vector) != dimension:
        st.error(f"Test vector must have {dimension} elements.")
    else:
        in_span, coeffs = is_in_span(test_vector, vectors)
        if in_span:
            st.success(f"✅ {test_vector.tolist()} is in the span of the input vectors.")
            st.markdown("Linear combination:")
            parts = [f"{round(c, 2)} × {v.tolist()}" for c, v in zip(coeffs, vectors)]
            st.latex(" + ".join(parts))
        else:
            st.error(f"❌ {test_vector.tolist()} is not in the span of the input vectors.")
