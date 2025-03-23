from manim import *
from manim.utils.color import ManimColor
import numpy as np
import matplotlib
from scipy.special import i0
from scipy.stats import vonmises


class MovingLines(Scene):
    def construct(self):
        # Initialize box
        box = Rectangle(height=1.5, width=10)#.move_to(np.array([-1, 0, 0]))
        
        # Initialize scale
        bottom_left = box.get_corner(DL)
        bottom_right = box.get_corner(DR)
        top_left = box.get_corner(UL)
        top_right = box.get_corner(UR)
        offset = DOWN * 0.2
        scale_left = bottom_left + offset
        scale_right = bottom_right + offset
        scale_len = bottom_right - bottom_left
        scale_line = Line(scale_left, scale_right)

        # Initialize ticks for scale
        major_tick_len = 0.3
        minor_tick_len = 0.15
        ticks = VGroup()
        labels = VGroup()
        for deg in range(0, 361, 10):
            fraction = deg / 360
            tick_position = scale_left + fraction * scale_len
            tick_len = minor_tick_len if deg % 30 else major_tick_len
            tick = Line(tick_position, tick_position + DOWN * tick_len)
            tick.cap_style = CapStyleType.ROUND
            ticks.add(tick)
            if deg % 30 == 0:
                label = MathTex(f"{deg}^\\circ").next_to(tick, DOWN, buff=0.1).scale(0.7)
                labels.add(label)

        # Load angle data (epochs x bars)
        bar_angles = np.load("stage1_angles.npy")
        epochs, num_bars = bar_angles.shape

        # Initialize colors
        max_bars = 64
        cmap = matplotlib.colormaps['rainbow']
        values = np.linspace(0, 1, max_bars)
        colors = [cmap(v) for v in values]
        colors = np.array(colors)

        # max_bars = 2
        # num_bars = 2
        # epochs = 3
        # bar_angles = np.array([[10, 40],
        #                        [330, 60],
        #                        [320, 70]
        #                        ])
        
        # A helper function to update bar positions from a continuous parameter.
        def get_updater(tracker):
            def update_bar(mob, dt):
                # Use modulo arithmetic to keep the bar within the box.
                frac = tracker.get_value() % 1  
                new_start = bottom_left + frac * scale_len
                new_end = top_left + frac * scale_len
                mob.put_start_and_end_on(new_start, new_end)
            return update_bar
        
        # Initialize bars with a ValueTracker controlling a continuous fraction parameter.
        bars = VGroup()
        for bar_id in range(num_bars):
            initial_angle = bar_angles[0][bar_id]
            initial_fraction = initial_angle / 360.0  # normalized initial fraction
            bar_start_position = bottom_left + initial_fraction * scale_len
            bar_end_position = top_left + initial_fraction * scale_len
            bar = Line(start=bar_start_position, end=bar_end_position).set_color(ManimColor(colors[bar_id]))
            # Each bar gets its own ValueTracker. The tracker may take on values outside [0,1]
            # but the updater always wraps it using modulo arithmetic.
            bar.param_tracker = ValueTracker(initial_fraction)
            bar.add_updater(get_updater(bar.param_tracker))
            bars.add(bar)
        
        # Initialize indicators
        epoch_text = Text("Epoch: ", font_size=20).next_to(box, UP, buff=0.3)
        epoch_indicator = Integer(number=0, font_size=30).next_to(epoch_text, RIGHT, buff=0.1)
        epoch_indicator.add_updater(lambda num: num.next_to(epoch_text, RIGHT, buff=0.1))
        
        # Create initial objects
        self.play(Create(box), run_time=1)
        self.play(Create(scale_line), run_time=1)
        self.play(Create(ticks), Create(labels), run_time=1)
        self.play(Create(bars), Write(epoch_text), Write(epoch_indicator), run_time=1)
        self.add(epoch_text, epoch_indicator)
        self.bring_to_front(box)
        
        # Animation Loop: for each epoch, update each bar's continuous parameter
        # by calculating the minimal (wrap-around) difference.
        for epoch in range(0, epochs):
            animations = []
            animations.append(Count(epoch_indicator, epoch, epoch+1))
            for bar_id, bar in enumerate(bars):
                # Get current displayed fraction (in [0,1])
                current_norm = bar.param_tracker.get_value() % 1
                # Compute the target fraction from the new angle.
                target_norm = ((bar_angles[epoch][bar_id] % 360) / 360.0)
                
                # Compute the difference (in normalized units) and adjust for wrap-around.
                diff = target_norm - current_norm
                if diff > 0.5:
                    diff -= 1
                elif diff < -0.5:
                    diff += 1
                
                # The new continuous parameter value.
                target_param = bar.param_tracker.get_value() + diff
                animations.append(bar.param_tracker.animate.set_value(target_param))
            self.play(*animations, run_time=1)
            self.wait(0.2)



def von_mises_pdf(theta, mu, kappa):
    return np.exp(kappa * np.cos(theta - mu)) / (2 * np.pi * i0(kappa))


class vMFDistributions(Scene):    
    def construct(self):
        equation = MathTex(
            r"p(x | \mu, \kappa) = C_p(\kappa) e^{\kappa \mu^T x}",
            r"\quad \text{where} \quad",
            r"C_p(\kappa) = \frac{\kappa^{p/2-1}}{(2\pi)^{p/2} I_{p/2-1}(\kappa)}"
        ).scale(0.7)

        # Position the equation in the center
        equation.arrange(DOWN, aligned_edge=LEFT)
        equation.move_to(4*LEFT)

        line_mu_start = 3.3*UP + 5*LEFT
        line_mu_end = 3.3*UP + 1*LEFT
        line_mu = Line(line_mu_start, line_mu_end)
        line_mu_start_mark = MathTex(r"{0}^\circ").next_to(line_mu_start, direction=DOWN, buff=0.15).scale(0.7)
        line_mu_end_mark = MathTex(r"{360}^\circ").next_to(line_mu_end, direction=DOWN, buff=0.15).scale(0.7)
        dot_mu_pos = line_mu_start
        indicator_mu_val = ValueTracker((dot_mu_pos[0] - line_mu_start[0]) / (line_mu_end[0] - line_mu_start[0]))
        dot_mu = Dot(dot_mu_pos, radius=0.1).set_color(RED)
        dot_mu.add_updater(lambda m: m.move_to(
            line_mu_start + (indicator_mu_val.get_value() / 360) * (line_mu_end - line_mu_start)
        ))
        text_mu = MathTex(r"\boldsymbol{\mu}").next_to(line_mu, direction=LEFT, buff=0.2).scale(0.7)

        line_kappa_start = 3.3*UP + 2*RIGHT
        line_kappa_end = 3.3*UP + 6*RIGHT
        line_kappa = Line(line_kappa_start, line_kappa_end)
        line_kappa_start_mark = MathTex("0").next_to(line_kappa_start, direction=DOWN, buff=0.15).scale(0.7)
        line_kappa_end_mark = MathTex("10").next_to(line_kappa_end, direction=DOWN, buff=0.15).scale(0.7)
        dot_kappa_pos = line_kappa_start
        indicator_kappa_val = ValueTracker((dot_kappa_pos[0] - line_kappa_start[0]) / (line_kappa_end[0] - line_kappa_start[0]))
        dot_kappa = Dot(dot_kappa_pos, radius=0.1).set_color(YELLOW)
        dot_kappa.add_updater(lambda m: m.move_to(
            line_kappa_start + (indicator_kappa_val.get_value() / 10) * (line_kappa_end - line_kappa_start)
        ))
        text_kappa = MathTex(r"\boldsymbol{\kappa}").next_to(line_kappa, direction=LEFT, buff=0.2).scale(0.7)


        num_circles = 3  # Number of concentric circles
        max_radius = 3    # Maximum radius of the largest circle
        circle_spacing = max_radius / num_circles
        circle_center = ORIGIN + DOWN/2 + RIGHT*3
        circles = VGroup()
        ticks = VGroup()
        labels_ang = VGroup()
        labels_dis = VGroup()
        for angle in range(0, 360, 45):
            rad = np.radians(angle)
            start = circle_center
            end = np.array([np.cos(rad) * max_radius, np.sin(rad) * max_radius, 0]) + circle_center
            tick = Line(start, end, color=GRAY_E)
            ticks.add(tick)

            # Position labels_ang slightly beyond the tick marks
            label_position = np.array([np.cos(rad) * (max_radius + 0.2), np.sin(rad) * (max_radius + 0.2), 0]) + circle_center
            label = MathTex(fr"{angle}^\circ").scale(0.5).set_color(GRAY_A).move_to(label_position)
            labels_ang.add(label)
        
        for i in range(num_circles):
            circle = Circle(radius=(i+1) * circle_spacing, color=GRAY_E, arc_center=circle_center)
            circles.add(circle)
            rad = np.radians(157.5)
            label_position = np.array([np.cos(rad) * (i+1.2), np.sin(rad) * (i+1.2), 0]) + circle_center
            label = MathTex(rf"{(i+1)/2}").scale(0.5).set_color(GRAY_A).move_to(label_position)
            labels_dis.add(label)


        length_mu_arrow = 2
        mu_arrow_start = circle_center
        mu_arrow_end = mu_arrow_start + length_mu_arrow*RIGHT
        mu_arrow = Arrow(start=mu_arrow_start, end=mu_arrow_end, stroke_width=5, buff=0).set_color(RED)
        mu_arrow.add_updater(lambda m: m.put_start_and_end_on(
            mu_arrow_start,
            mu_arrow_start + length_mu_arrow*np.array([
                np.cos(np.deg2rad(indicator_mu_val.get_value())),
                np.sin(np.deg2rad(indicator_mu_val.get_value())),
                0
            ])
        ))

        num_points = 500
        scale_factor = 2
        polar_curve = VMobject()
        polar_curve.set_color(YELLOW)
        def update_curve(mob: VMobject):
            mu = indicator_mu_val.get_value() * np.pi / 180
            kappa = indicator_kappa_val.get_value()
            theta_vals = np.linspace(0, 2 * np.pi, num_points)
            pdf_vals = von_mises_pdf(theta_vals, mu, kappa)
            points = [
                mu_arrow_start + scale_factor * pdf * np.array([np.cos(theta), np.sin(theta), 0])
                for theta, pdf in zip(theta_vals, pdf_vals)
            ]
            mob.set_points_smoothly(points)
        polar_curve.add_updater(update_curve)

        delta_min = 30
        delta_top = ValueTracker(180.0)
        delta_top.add_updater(lambda m: m.set_value(
            180 - (indicator_kappa_val.get_value() / 10) * (180 - delta_min)
        ))
        self.add(delta_top)

        # def arrow_sample_updater(idx, num_arrow_samples, arrow_start):
        #     def update(arrow: Arrow):
        #         delta_top_val = delta_top.get_value()
        #         mu_val = indicator_mu_val.get_value()
        #         # print(delta_top_val)
        #         num_arrow_samples_per_side = (num_arrow_samples + 1) // 2
        #         delta_step = delta_top_val/num_arrow_samples_per_side
        #         # which_side = idx if idx<num_arrow_samples//2 else np.abs(num_arrow_samples - idx)
        #         if idx < (num_arrow_samples//2+1):
        #             step_idx = idx
        #             angle_arrow = delta_top_val - delta_step * step_idx + delta_step/2 + mu_val
        #         else:
        #             step_idx = num_arrow_samples+1 - idx
        #             angle_arrow = -delta_top_val + delta_step * step_idx - delta_step/2 + mu_val
        #         new_end = arrow_start + arrow_sample_length*np.array([
        #                     np.cos(np.deg2rad(angle_arrow)),
        #                     np.sin(np.deg2rad(angle_arrow)),
        #                     0
        #         ])
                
        #         arrow.put_start_and_end_on(arrow_start, new_end)
        #     return update


        # Instead of equally spacing arrows linearly, we sample from the von Mises distribution.
        # Here we use the ppf (inverse CDF) to determine arrow angles.
        def arrow_sample_updater(idx, num_arrow_samples, arrow_start):
            def update(arrow: Arrow):
                # Compute symmetric quantile for arrow idx.
                # idx runs from 1 to num_arrow_samples.
                p = 0.5 + (idx - 1 - (num_arrow_samples - 1) / 2) / num_arrow_samples
                # Get μ in radians.
                mu_rad = indicator_mu_val.get_value() * np.pi / 180
                # κ value.
                kappa = indicator_kappa_val.get_value()
                # For kappa==0, the vonmises distribution is uniform.
                if kappa == 0:
                    angle_rad = mu_rad + 2 * np.pi * (p - 0.5)
                else:
                    angle_rad = vonmises.ppf(p, kappa, loc=mu_rad)
                angle_deg = np.rad2deg(angle_rad)
                new_end = arrow_start + arrow_sample_length * np.array([
                    np.cos(np.deg2rad(angle_deg)),
                    np.sin(np.deg2rad(angle_deg)),
                    0
                ])
                arrow.put_start_and_end_on(arrow_start, new_end)
            return update


        arrow_sample_group = VGroup()
        arrow_sample_length = 1
        num_arrow_samples = 20
        # num_arrow_samples_per_side = (num_arrow_samples + 1) // 2 + 1
        for i in range(num_arrow_samples):
            arrow_sample_start = mu_arrow_start
            arrow_sample_end = arrow_sample_start + arrow_sample_length*RIGHT
            arrow_tmp = Arrow(arrow_sample_start, arrow_sample_end, stroke_width=1)
            arrow_tmp.add_updater(arrow_sample_updater(i+1, num_arrow_samples, arrow_sample_start))
            arrow_sample_group.add(arrow_tmp)

        indicator_kappa_val.set_value(4)
        self.play(Write(equation))
        self.play(*[GrowFromCenter(circle) for circle in circles], Create(ticks), Create(labels_ang), Create(labels_dis))
        self.play(Create(line_mu), Create(line_mu_start_mark), Create(line_mu_end_mark), Create(text_mu), Create(dot_mu),
                  Create(line_kappa), Create(line_kappa_start_mark), Create(line_kappa_end_mark), Create(text_kappa), Create(dot_kappa),
                  Create(mu_arrow), Create(polar_curve), Create(arrow_sample_group))
        # self.add(line_mu, line_mu_start_mark, line_mu_end_mark, text_mu, dot_mu)
        # self.add(line_kappa, line_kappa_start_mark, line_kappa_end_mark, text_kappa, dot_kappa)
        # self.add(mu_arrow, polar_curve, arrow_sample_group)
        self.play(indicator_mu_val.animate.set_value(360), run_time=3, rate_func=linear)
        self.play(indicator_mu_val.animate.set_value(0), run_time=3, rate_func=linear)
        self.play(indicator_kappa_val.animate.set_value(10), run_time=3, rate_func=linear)
        self.play(indicator_kappa_val.animate.set_value(0), run_time=3, rate_func=linear)
        self.play(indicator_kappa_val.animate.set_value(4), run_time=3, rate_func=linear)
        # self.wait(2)