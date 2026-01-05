"""
问题2：使用遗传算法优化飞行参数
功能：在已知的好点周围生成初始种群，进行局部搜索优化
策略：在所有变量的已知好点周围生成种子，进行局部搜索优化
"""

from ClassFly import *
from ClassMission import *
from ClassSmoke import *
from clc import *
from genetic_algorithm import GeneticAlgorithm


def q2():
    """
    使用自定义初始种群进行遗传算法优化
    
    策略说明：
    - 在所有变量的已知好点周围生成初始种子
    - 基于已知的接近最优解的参数值进行局部搜索优化
    """
    from genetic_algorithm import create_seeds_around_points
    
    # 输出标题信息
    print("=" * 60)
    print("示例6d: 在指定点周围生成随机种子")
    print("=" * 60)
    
    # 用于跟踪适应度函数调用次数
    fitness_call_count = [0]  # 使用列表以便在嵌套函数中修改
    
    def fitness_function(vars_dict):
        """
        适应度函数：计算给定参数下的遮蔽时间
        
        参数:
            vars_dict: 包含所有优化变量的字典
                - fly1_angle: 飞行角度
                - fly1_v: 飞行速度
                - smoke1_1_fly_time: 烟雾1的飞行时间
                - smoke1_1_falltime: 烟雾1的降落时间
        
        返回:
            total_time[0]: 计算得到的遮蔽时间（适应度值，越大越好，最大化问题）
        """
        fitness_call_count[0] += 1
        if fitness_call_count[0] % 5 == 0:
            print(f"  正在计算第 {fitness_call_count[0]} 个个体...", end="\r", flush=True)
        
        # 初始化系统状态
        init()
        # 设置飞行参数
        smoke_class_list[0].flag = 1
        mission_class_list[0].flag = 1

        fly1.angle = vars_dict["fly1_angle"]
        fly1.v = vars_dict["fly1_v"]
        # 设置烟雾参数
        smoke_class_list[0].fly_time = vars_dict["smoke1_1_fly_time"]
        smoke_class_list[0].fall_time = vars_dict["smoke1_1_falltime"]
        # 执行计算
        clc()
        # 返回遮蔽时间作为适应度值（最大化问题）
        return total_time[0]
    
    # ==================== 变量范围定义 ====================
    # 定义每个优化变量的搜索范围（最小值, 最大值）
    variable_ranges = {
        "fly1_angle": (170, 180),          # 飞行角度范围（度）
        "fly1_v": (70, 140),                # 飞行速度范围
        "smoke1_1_fly_time": (0, 10),      # 烟雾1飞行时间范围
        "smoke1_1_falltime": (0, 10)       # 烟雾1降落时间范围
    }
    
    # ==================== 生成初始种群 ====================
    # 策略：在所有变量的已知好点周围生成种子
    # 定义已知的好点（基于q1.py中的接近最优解的参数值）
    centers = [
        {
            "fly1_angle": 180,              # 飞行角度
            "fly1_v": 120,                  # 飞行速度
            "smoke1_1_fly_time": 1.5,       # 烟雾1飞行时间
            "smoke1_1_falltime": 3.6        # 烟雾1降落时间
        }
    ]

    # 在已知好点周围生成初始种子
    initial_seeds = create_seeds_around_points(
        ["fly1_angle", "fly1_v", "smoke1_1_fly_time", "smoke1_1_falltime"],  # 所有变量
        centers,                  # 中心点列表
        n_per_center=20,          # 每个中心点生成20个随机种子
        noise_scale=0.5           # 噪声比例：在中心点值的基础上添加高斯噪声
    )
    
    # ==================== 遗传算法参数配置 ====================
    # 可以在这里调整遗传算法的关键参数以优化性能和结果
    
    # 种群参数
    POPULATION_SIZE = 200          # 种群大小（每代的个体数量，越大搜索范围越广但计算越慢）
    MAX_GENERATIONS = 20          # 最大迭代代数（越多可能找到更好的解但计算越慢）
    ELITISM_COUNT = 2             # 精英保留数量（每代保留的最优个体数，保持种群多样性）
    
    # 遗传操作参数
    CROSSOVER_RATE = 0.8          # 交叉率（0-1，越高越容易产生新个体，推荐0.7-0.9）
    MUTATION_RATE = 0.2           # 变异率（0-1，越高越容易跳出局部最优，推荐0.05-0.2）
    
    # 其他参数
    VERBOSE = True                # 是否显示优化过程信息
    MAXIMIZE = True               # 是否为最大化问题
    
    # ==================== 运行遗传算法 ====================
    print(f"\n{'='*60}")
    print(f"遗传算法参数配置:")
    print(f"  种群大小: {POPULATION_SIZE}")
    print(f"  最大代数: {MAX_GENERATIONS}")
    print(f"  精英保留: {ELITISM_COUNT}")
    print(f"  交叉率: {CROSSOVER_RATE}")
    print(f"  变异率: {MUTATION_RATE}")
    print(f"{'='*60}")
    print(f"注意：适应度函数计算较慢（每个个体可能需要几分钟），请耐心等待...\n")
    
    ga = GeneticAlgorithm(
        variable_ranges=variable_ranges,      # 变量范围定义
        fitness_function=fitness_function,    # 适应度函数
        population_size=POPULATION_SIZE,      # 种群大小
        max_generations=MAX_GENERATIONS,      # 最大迭代代数
        crossover_rate=CROSSOVER_RATE,       # 交叉率
        mutation_rate=MUTATION_RATE,          # 变异率
        elitism_count=ELITISM_COUNT,          # 精英保留数量
        initial_population=initial_seeds,    # 自定义初始种群（20个种子）
        verbose=VERBOSE,                      # 显示优化过程信息
        maximize=MAXIMIZE                     # 最大化问题
    )
    
    # 运行遗传算法，返回最优个体和最优适应度值
    best, fitness = ga.run()
    
    # 输出优化结果
    print(f"最优解: fly1_angle={best['fly1_angle']:.6f}, "
          f"fly1_v={best['fly1_v']:.6f}, "
          f"smoke1_1_fly_time={best['smoke1_1_fly_time']:.6f}, "
          f"smoke1_1_falltime={best['smoke1_1_falltime']:.6f}, "
          f"适应度: {fitness:.6f}")
    print(f"生成了 {len(initial_seeds)} 个初始种子\n")


if __name__ == "__main__":
    # 直接运行脚本时执行优化
    q2()