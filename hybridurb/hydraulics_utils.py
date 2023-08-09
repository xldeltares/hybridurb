import numpy as np


class HydraulicUtils:
    @staticmethod
    def calculate_circle_perimeter(radius: float) -> float:
        # Calculate the perimeter of the circle
        perimeter = 2 * np.pi * radius
        return perimeter

    @staticmethod
    def calculate_rectangle_perimeter(height: float, width: float) -> float:
        # Calculate the perimeter of the rectangle
        perimeter = 2 * height + width
        return perimeter

    @staticmethod
    def calculate_circle_area(radius: float) -> float:
        # Calculate the area of the circle
        area = np.pi * radius**2
        return area

    @staticmethod
    def calculate_rectangle_area(height: float, width: float) -> float:
        # Calculate the area of the rectangle
        area = height * width
        return area

    @staticmethod
    def calculate_zw_area(levels: np.ndarray, flow_widths: np.ndarray) -> float:
        """
        Calculate the area for 'zw' type crosssections.

        Parameters:
        levels (numpy.ndarray): Array of levels.
        flow_widths (numpy.ndarray): Array of flow widths.

        Returns:
        float: Array of areas.
        """
        area = np.trapz(flow_widths, x=levels)
        return float(area)

    @staticmethod
    def calculate_zw_perimeter(
        levels: np.ndarray, flow_widths: np.ndarray, closed: bool
    ) -> float:
        """
        Calculate the perimeter for 'zw' type crosssections.

        Parameters:
        levels (numpy.ndarray): Array of levels.
        flow_widths (numpy.ndarray): Array of flow widths.
        closed (bool): Whether the crosssection is closed.

        Returns:
        float: perimeter.
        """
        # Check if levels contains 0 and remove it
        if 0 in levels:
            zero_index = np.where(levels == 0)[0]
            levels = np.delete(levels, zero_index)
            flow_widths = np.delete(flow_widths, zero_index)

        # Calculate perimeter
        perimeter = np.trapz(flow_widths, x=levels) / (
            (flow_widths[-1] - flow_widths[0])
            if closed
            else np.trapz(flow_widths / levels, x=levels)
        )
        return float(perimeter)

    @staticmethod
    def calculate_hydraulic_diameter(area: float, perimeter: float) -> float:
        # Calculate the hydraulic diameter
        hydraulic_diameter = 4 * area / perimeter
        return hydraulic_diameter

    @staticmethod
    def calculate_velocity(
        hydraulic_diameter: float,
        hydraulic_gradient: float,
        roughness_coefficient: float,
    ) -> float:
        """
        Function to calculate the velocity of a fluid in a pipe using the Colebrook-White equation.

        Parameters:
        hydraulic_diameter (float): Hydraulic diameter of the pipe, in m
        hydraulic_gradient (float): Hydraulic gradient (change in head loss per unit length), in m/m
        roughness_coefficient (float): Roughness coefficient of the pipe, in mm

        Returns:
        velocity (float): Velocity of the fluid in the pipe, in m/s
        """
        if hydraulic_gradient == 0.0:
            return 0.0

        # Check if the hydraulic gradient is negative
        negative_gradient = False
        if hydraulic_gradient < 0:
            hydraulic_gradient = np.absolute(hydraulic_gradient)
            negative_gradient = True

        # Calculate the velocity using the Colebrook-White equation
        velocity = (
            -2.0
            * (2.0 * 9.807 * hydraulic_diameter * hydraulic_gradient) ** (1.0 / 2.0)
            * np.log10(
                roughness_coefficient * (10.0 ** (-3.0)) / (3.7 * hydraulic_diameter)
                + (2.51 * 10.0 ** (-6.0))
                / (
                    hydraulic_diameter
                    * (2.0 * 9.807 * hydraulic_diameter * hydraulic_gradient)
                    ** (1.0 / 2.0)
                )
            )
        )

        # If the hydraulic gradient was negative, the velocity is also negative
        if negative_gradient:
            velocity = -1.0 * velocity

        return velocity

    @staticmethod
    def calculate_capacity(velocity: float, flowarea: float) -> float:
        # Function to calculate capcity
        capcity = velocity * flowarea
        return capcity

    @staticmethod
    def calculate_reynolds_number(velocity: float, hydraulic_diameter: float) -> float:
        """
        Function to calculate the Reynolds number for flow in a pipe.

        Parameters:
        velocity (float): Velocity of the fluid in the pipe, in m/s
        hydraulic_diameter (float): Hydraulic diameter of the pipe, in m

        Returns:
        reynolds_number (float): Reynolds number, dimensionless
        """

        # Calculate the Reynolds number
        reynolds_number = velocity * hydraulic_diameter / 10 ** (-6)

        return reynolds_number

    @staticmethod
    def calculate_friction_factor(
        reynolds_number: float, roughness_coefficient: float, hydraulic_diameter: float
    ) -> float:
        """
        Function to calculate the Darcy friction factor for flow in a pipe.

        Parameters:
        reynolds_number (float): Reynolds number, dimensionless
        roughness_coefficient(float): Roughness coefficient of the pipe, in mm
        hydraulic_diameter(float): Hydraulic diameter of the pipe, in m

        Returns:
        friction_factor(float): Darcy friction factor, dimensionless
        """
        # FIXME: check this
        roughness_coefficient = roughness_coefficient * (10.0 ** (-3.0))

        # Calculate the Darcy friction factor
        friction_factor = 0.25 / (
            np.log10(
                roughness_coefficient / (3.7 * hydraulic_diameter)
                + 5.74 / (reynolds_number**0.9)
            )
            ** 2
        )

        return friction_factor

    @staticmethod
    def calculate_head_loss(
        friction_factor: float,
        velocity: float,
        pipe_length: float,
        hydraulic_diameter: float,
    ) -> float:
        """
        Function to calculate the Darcy-Weisbach head loss in a pipe.

        Parameters:
        friction_factor (float): Darcy friction factor, dimensionless
        velocity (float): Velocity of the fluid in the pipe, in m/s
        pipe_length (float): Length of the pipe, in m
        hydraulic_diameter (float): Hydraulic diameter of the pipe, in m

        Returns:
        head_loss (float): Head loss due to friction, in m
        """
        # Calculate the head loss
        head_loss = (
            friction_factor
            * (pipe_length / hydraulic_diameter)
            * (velocity**2 / (2 * 9.807))
        )

        return head_loss
