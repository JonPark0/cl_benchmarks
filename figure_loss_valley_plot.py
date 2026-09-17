import numpy as np
import matplotlib.pyplot as plt

# 논문용 폰트 크기 설정
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14})

# 2x4 레이아웃 설정 (가로 크기를 24로 늘림)
fig = plt.figure(figsize=(24, 10))

# 2차원 XY 평면 격자 생성
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# 1~3열: 가상의 손실 함수 정의 (아래로 파인 계곡 형태)
Z_A = 2 - 2 * np.exp(-((X + 1)**2 + (Y - 1)**2) / 1.5)
Z_B = 2 - 2 * np.exp(-((X - 1)**2 + (Y + 1)**2) / 1.5)
Z_AB = Z_A + Z_B

# 4열: 겹치는 부분(간섭 영역)을 솟아오른 모형(Peak)으로 수학적 모델링
Influence_A = np.exp(-((X + 1)**2 + (Y - 1)**2) / 1.5)
Influence_B = np.exp(-((X - 1)**2 + (Y + 1)**2) / 1.5)
Z_Overlap = Influence_A * Influence_B
# 시각적으로 뚜렷하게 솟아오르도록 정규화 및 높이(3.5) 스케일링
Z_Overlap = (Z_Overlap / np.max(Z_Overlap)) * 3.5

# ----------------------------------------------------
# 상단 (Row 1): 2D 등고선 (Contour)
# ----------------------------------------------------
ax1 = fig.add_subplot(2, 4, 1)
c1 = ax1.contourf(X, Y, Z_A, levels=20, cmap='viridis', alpha=0.9)
ax1.set_title('Col 1: Task A Loss (Valley)')
ax1.set_xlabel('X') ; ax1.set_ylabel('Y')

ax2 = fig.add_subplot(2, 4, 2)
c2 = ax2.contourf(X, Y, Z_B, levels=20, cmap='viridis', alpha=0.9)
ax2.set_title('Col 2: Task B Loss (Valley)')
ax2.set_xlabel('X') ; ax2.set_ylabel('Y')

ax3 = fig.add_subplot(2, 4, 3)
c3 = ax3.contourf(X, Y, Z_AB, levels=40, cmap='viridis', alpha=0.9)
ax3.set_title('Col 3: A + B Combined')
ax3.set_xlabel('X') ; ax3.set_ylabel('Y')

# 4번째 열: 간섭 강도 
ax4 = fig.add_subplot(2, 4, 4)
c4 = ax4.contourf(X, Y, Z_Overlap, levels=20, cmap='Reds', alpha=0.9)
ax4.set_title('Col 4: Interference Magnitude (Peak)')
ax4.set_xlabel('X') ; ax4.set_ylabel('Y')
plt.colorbar(c4, ax=ax4, fraction=0.046, pad=0.04)

# ----------------------------------------------------
# 하단 (Row 2): 3D 표면 (Surface)
# ----------------------------------------------------
z_min_loss, z_max_loss = 0, 4

ax5 = fig.add_subplot(2, 4, 5, projection='3d')
ax5.plot_surface(X, Y, Z_A, cmap='viridis', edgecolor='none', alpha=0.8)
ax5.set_title('3D: Task A')
ax5.set_zlim(z_min_loss, z_max_loss)

ax6 = fig.add_subplot(2, 4, 6, projection='3d')
ax6.plot_surface(X, Y, Z_B, cmap='viridis', edgecolor='none', alpha=0.8)
ax6.set_title('3D: Task B')
ax6.set_zlim(z_min_loss, z_max_loss)

ax7 = fig.add_subplot(2, 4, 7, projection='3d')
ax7.plot_surface(X, Y, Z_AB, cmap='viridis', edgecolor='none', alpha=0.8)
ax7.set_title('3D: A + B')
ax7.set_zlim(z_min_loss, z_max_loss)

ax8 = fig.add_subplot(2, 4, 8, projection='3d')
ax8.plot_surface(X, Y, Z_Overlap, cmap='Reds', edgecolor='none', alpha=0.9)
ax8.set_title('3D: Interference Magnitude')
ax8.set_zlim(0, 4) 

# 3D 그래프 시점 및 축 설정
for ax in [ax5, ax6, ax7, ax8]:
    ax.view_init(elev=35, azim=45)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # [추가됨] 3D 그래프의 X축과 Y축 반전
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    if ax == ax8:
        ax.set_zlabel('Magnitude')
    else:
        ax.set_zlabel('Loss')

plt.tight_layout()
plt.show()