template<typename T>
static void put(std::fstream &f, const T value) {
    union {
        char buf[sizeof(T)];
        T val;
    } helper;
    helper.val = value;
    std::reverse(helper.buf, helper.buf + sizeof(T));
    f.write(helper.buf, sizeof(T));
}

template <typename real, int time_order, class Problem>
void Solver<real, time_order, Problem>::save(const std::string &prefix) {
    std::string path(prefix + "." + std::to_string(step()) + ".vtk");
    std::fstream f(path, std::ios::out | std::ios::binary);
    if (!f) {
        std::cerr << "Unable to open file `" << prefix << "' for writing" << std::endl;
        return;
    }

    u_host = u;
    b_host = b;

    f << "# vtk DataFile Version 3.0\n";
    f << "Shallow water solver output\n";
    f << "BINARY\n";
    f << "DATASET RECTILINEAR_GRID\n";
    f << "DIMENSIONS " << M + 1 << " " << N + 1 << " 1";
    f << "\nX_COORDINATES " << M + 1 << " float\n";
    for (int i = 0; i <= M; i++)
        put<float>(f, i * hx);
    f << "\nY_COORDINATES " << N + 1 << " float\n";
    for (int j = 0; j <= N; j++)
        put<float>(f, j * hy);
    f << "\nZ_COORDINATES " << 1 << " float\n";
    put<float>(f, 0);
    f << "\nCELL_DATA " << N * M;

    f << "\nSCALARS h float\nLOOKUP_TABLE default\n";
    for (int j = 1; j <= N; j++)
        for (int i = 1; i <= M; i++)
            put<float>(f, u_host.h(i, j).v);

    f << "\nSCALARS b float\nLOOKUP_TABLE default\n";
    for (int j = 1; j <= N; j++)
        for (int i = 1; i <= M; i++)
            put<float>(f, b_host(i, j).v);

    f << "\nSCALARS zeta float\nLOOKUP_TABLE default\n";
    for (int j = 1; j <= N; j++)
        for (int i = 1; i <= M; i++)
            put<float>(f, b_host(i, j).v + u_host.h(i, j).v);

    f << "\nVECTORS v float\n";
        for (int j = 1; j <= N; j++)
            for (int i = 1; i <= M; i++) {
                real h = u_host.h(i, j).v;
                real vx = u_host.hu(i, j).v / h;
                real vy = u_host.hv(i, j).v / h;
                put<float>(f, vx);
                put<float>(f, vy);
                put<float>(f, 0);
            }
    f << std::endl;
    f.close();
}
