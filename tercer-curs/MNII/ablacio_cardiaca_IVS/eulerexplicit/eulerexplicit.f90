program calorEE
    implicit none

    ! parametres fisics
    real(8), parameter :: L = 0.02d0
    real(8), parameter :: k = 0.56d0
    real(8), parameter :: rho = 1081.d0
    real(8), parameter :: cv = 3686.d0
    real(8), parameter :: Pext = 9.44d5
    real(8), parameter :: Tref = 309.65d0
    real(8), parameter :: deltaT = 43.5d0
    real(8), parameter :: Ttarget = 353.15d0     ! 80 C
    real(8), parameter :: Tsafe = 323.15d0       ! 50 C
    real(8), parameter :: t0 = (L*L*rho*cv)/k

    ! parametre adimensional
    real(8), parameter :: C = Pext*L*L/(k*deltaT)

    ! parametres numerics
    integer, parameter :: Nx = 101
    real(8), parameter :: dx = L/(Nx-1)
    real(8) :: dx_star, dt_star, dt
    integer, parameter :: max_steps = 2000000
    integer :: i_sick_start, i_sick_end
    real(8) :: r, Tmax, Tmax_sick, Tmax_healthy
    real(8) :: t_dim, t_sick, Tloc
    integer :: i, step

    real(8), dimension(Nx) :: theta, theta_new

    dx_star = dx / L
    dt_star = 0.25d0 * dx_star**2
    dt = dt_star * t0
    r = dt_star / (dx_star**2)

    theta = 0.d0

    ! zona malalta
    i_sick_start = nint(0.0075d0 / L * (Nx-1)) + 1
    i_sick_end   = nint(0.0125d0 / L * (Nx-1)) + 1

    t_sick = 0.d0

    do step = 1, max_steps

        theta_new = theta

        ! euler explicit
        do i = 2, Nx-1
            theta_new(i) = theta(i) + r * (theta(i-1) - 2.d0*theta(i) + theta(i+1)) + C*dt_star
        end do

        ! condicions de contorn
        theta_new(1) = 0.d0
        theta_new(Nx) = 0.d0

        theta = theta_new

        ! control de condicions
        Tmax = maxval(theta)*deltaT + Tref
        Tmax_healthy = maxval(theta(1:i_sick_start-1))*deltaT + Tref
        Tmax_healthy = max(Tmax_healthy, &
                           maxval(theta(i_sick_end+1:Nx))*deltaT + Tref)

        if (Tmax >= Ttarget) exit   ! limit 80 °C

        if (Tmax_healthy >= Tsafe) exit   ! limit 50°C zona sana

        ! temps zona malalta 50–80°C
        do i = i_sick_start, i_sick_end
            Tloc = theta(i)*deltaT + Tref
            if (Tloc >= Tsafe .and. Tloc <= Ttarget) then
                t_sick = t_sick + dt
                exit
            end if
        end do

    end do

    t_dim = (step-1) * dt
    Tmax_sick = maxval(theta(i_sick_start:i_sick_end))*deltaT + Tref
    Tmax_healthy = max(maxval(theta(1:i_sick_start-1)), &
                       maxval(theta(i_sick_end+1:Nx)))*deltaT + Tref

    print *, "temps total", t_dim, "s"
    print *, "temps zona malalta entre 50-80C:", t_sick, "s"
    print *, "Tmax final:", Tmax, "K"
    print *, "Tmax malalta:", Tmax_sick, "K"
    print *, "Tmax sana:", Tmax_healthy, "K"
    print *, "posicio (m)   temperatura (K)"

    do i = 1, Nx
        print *, (i-1)*L/(Nx-1), theta(i)*deltaT + Tref
    end do

end program calorEE
