from ClassFly import *
from ClassMission import *
from ClassSmoke import *
from clc import *
from genetic_algorithm import GeneticAlgorithm

def q2():
    """使用自定义初始种群 - 方式4：在指定点周围生成随机种子"""
    from genetic_algorithm import create_seeds_around_points
    
    print("=" * 60)
    print("示例6d: 在指定点周围生成随机种子")
    print("=" * 60)
    
    def fitness_function(vars_dict):
        init()
        fly1.angle = vars_dict["fly1_angle"]
        fly1.v = vars_dict["fly1_v"]
        smoke_class_list[0].fly_time = vars_dict["smoke1_1_fly_time"]
        smoke_class_list[0].fall_time = vars_dict["smoke1_1_falltime"]
        clc()
        return total_time[0]
    
    # 方式4：在已知的好点周围生成随机种子（适合局部搜索）
    centers = [
        {"fly1_angle": 179 },      # 接近最优解
        {"fly1_angle": 175},    # 另一个好点
    ]

    initial_seeds = create_seeds_around_points(
        ["fly1_angle"],
        centers,
        n_per_center=20,  # 每个中心点生成20个随机种子
        noise_scale=0.05  # 噪声比例
    )
    
    ga = GeneticAlgorithm(
        variable_ranges={"fly1_angle": (170, 180), "fly1_v": (70, 140),"smoke1_1_fly_time":(0,10),"smoke1_1_falltime":(0,10)},
        fitness_function=fitness_function,
        population_size=50,
        max_generations=100,
        initial_population=initial_seeds,
        verbose= True
    )
    
    best, fitness = ga.run()
    print(f"最优解: fly1_angle={best['fly1_angle']:.6f}, fly1_v={best['fly1_v']:.6f},smoke1_1_fly_time={best['smoke1_1_fly_time']:.6f},smoke1_1_falltime={best['smoke1_1_falltime']:.6f}, 适应度: {fitness:.6f}")
    print(f"生成了 {len(initial_seeds)} 个初始种子\n")

if __name__ == "__main__":
    q2()