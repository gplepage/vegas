from cffi import FFI
ffibuilder = FFI()

# specify functions, etc made available to Python
ffibuilder.cdef("""
    double fcn(double x[], int dim);
    void batch_fcn(double ans[], double x[], int n, int dim);
    """)

# specify code needed to build the module
ffibuilder.set_source(
    module_name="cfcn_cffi",            # Python extension module (output)
    source="""
    // code for module
    double fcn(double x[], int dim);

    void batch_fcn(double ans[], double x[], int n, int dim)
    {
        int i;
        for(i=0; i<n; i++)
            ans[i] = fcn(&x[i * dim], dim);
    }
    """,
    sources=['cfcn.c'],     # other sources -- file containing fcn(x, dim)
    libraries=['m'],        # may need to specify the math library (-lm)
    )

if __name__ == "__main__":
    # create C code for module and compile it
    ffibuilder.compile(verbose=False)
