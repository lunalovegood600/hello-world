# hello-world
chuxuezhe
self.g_a = nn.Sequential(
    conv(3, N, kernel_size=3),
    conv(N, N, kernel_size=3, stride=2),
    GDN(N),
    conv(N, N, kernel_size=3),
    conv(N, N, kernel_size=3, stride=2),
    GDN(N),
    Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
    conv(N, N, kernel_size=3),
    conv(N, N, kernel_size=3, stride=2),
    GDN(N),
    conv(N, N, kernel_size=3),
    conv(N, M, kernel_size=3, stride=2),
    Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
)
self.g_s = nn.Sequential(
    Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
    deconv(M, N, kernel_size=3),
    deconv(N, N, kernel_size=3, stride=2),
    GDN(N, inverse=True),
    deconv(N, N, kernel_size=3),
    deconv(N, N, kernel_size=3, stride=2),
    GDN(N, inverse=True),
    Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
    deconv(N, N, kernel_size=3),
    deconv(N, N, kernel_size=3, stride=2),
    GDN(N, inverse=True),
    deconv(N, N, kernel_size=3),
    deconv(N, 3, kernel_size=3, stride=2),
)
