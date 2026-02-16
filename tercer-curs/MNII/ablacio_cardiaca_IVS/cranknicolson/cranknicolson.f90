program calorCN
    implicit none

    ! parametres fisics
    real(8), parameter :: L = 0.02d0
    real(8), parameter :: k = 0.56d0
    real(8), parameter :: rho = 1081.d0
    real(8), parameter :: cv = 3686.d0
    real(8), parameter :: Pext = 9.44d5
    real(8), parameter :: Tref = 309.65d0
    real(8), parameter :: deltaT = 43.5d0
    real(8), parameter :: Ttarget = 353.15d0        
    real(8), parameter :: Tsafe = 323.15d0          
    real(8), parameter :: t0 = (L*L*rho*cv)/k

    ! escalar adimensional
    real(8), parameter :: C = Pext*L*L/(k*deltaT)

    ! parametres numerics
    integer, parameter :: Nx = 101
    real(8), parameter :: dx = L/(Nx-1)
    real(8) :: dx_star, dt_star, dt
    integer, parameter :: max_steps = 2000000
    integer :: i_sick_start, i_sick_end

    real(8) :: r, Tmax, t_dim, t_sick, Tloc
    real(8) :: Tmax_healthy, Tmax_sick
    integer :: i, step

    ! arrays de Crank-Nicolson
    real(8), dimension(Nx) :: theta, theta_new, theta_prev
    real(8), dimension(Nx) :: aa, bb, cc, dd

    dx_star = dx / L
    dt_star = dx_star**2
    dt = dt_star * t0
    r = dt_star / (2.d0 * dx_star**2)

    theta = 0.d0

    ! localitzacio zona malalta
    i_sick_start = nint(0.0075d0 / L * (Nx-1)) + 1
    i_sick_end   = nint(0.0125d0 / L * (Nx-1)) + 1

    ! inicialitzem matriu
    do i = 1, Nx
        aa(i) = 0.d0
        bb(i) = 1.d0
        cc(i) = 0.d0
    end do

    do i = 2, Nx-1
        aa(i) = -r
        bb(i) = 1.d0 + 2.d0*r
        cc(i) = -r
    end do

    t_sick = 0.d0

    ! bucle temporal
    do step = 1, max_steps

        theta_prev = theta

        ! definim el vector d

        ! 0 en 1 i N per condicions de contorn
        dd(1) = 0.d0
        dd(Nx) = 0.d0

        do i = 2, Nx-1
            dd(i) = r*theta(i-1) + (1.d0 - 2.d0*r)*theta(i) + r*theta(i+1) + C*dt_star
        end do

        call thomas(aa, bb, cc, dd, theta_new, Nx)
        theta = theta_new

        Tmax = maxval(theta)*deltaT + Tref

        ! control 80C
        if (Tmax >= Ttarget) then
            theta = theta_prev
            exit
        end if

        ! control 50C regio sana
        Tmax_healthy = maxval(theta(1:i_sick_start-1))*deltaT + Tref
        Tmax_healthy = max(Tmax_healthy, maxval(theta(i_sick_end+1:Nx))*deltaT + Tref)

        if (Tmax_healthy >= Tsafe) then
            theta = theta_prev
            exit
        end if

        ! temps en que malalta 50-80
        do i = i_sick_start, i_sick_end
            Tloc = theta(i)*deltaT + Tref
            if (Tloc >= Tsafe .and. Tloc <= Ttarget) then
                t_sick = t_sick + dt
                exit   
            end if
        end do

    end do

    ! calculem el valor de diverses variables
    ! quan s'acaba la simulacio
    Tmax = maxval(theta)*deltaT + Tref
    t_dim = (step-1) * dt

    Tmax_sick = maxval(theta(i_sick_start:i_sick_end))*deltaT + Tref
    Tmax_healthy = max( maxval(theta(1:i_sick_start-1)), &
                             maxval(theta(i_sick_end+1:Nx)) )*deltaT + Tref


    print *, "temps total", t_dim, "s"
    print *, "temps amb zona malalta entre 50-80C:", t_sick, "s"
    print *, "temperatura maxima final:", Tmax, "K"
    print *, "temperatura maxima zona malalta:", Tmax_sick, "K"
    print *, "temperatura maxima zona sana:", Tmax_healthy, "K"
    print *, "posicio (m)   temperatura (K)"

    do i = 1, Nx
        print *, (i-1)*L/(Nx-1), theta(i)*deltaT + Tref
    end do

end program calorCN


! rutina thomas per resoldre sistemes tridiagonals
subroutine thomas(a, b, c, d, x, n)
    implicit none
    integer, intent(in) :: n
    real(8), dimension(n), intent(in) :: a, b, c
    real(8), dimension(n), intent(in) :: d
    real(8), dimension(n), intent(out) :: x

    real(8), dimension(n) :: cp, dp
    real(8) :: m
    integer :: i

    ! convertim el sistema en un triangular superior
    ! amb aquestes transformacions

    cp(1) = c(1) / b(1)
    dp(1) = d(1) / b(1)

    do i = 2, n
        m = b(i) - a(i)*cp(i-1)
        cp(i) = c(i) / m
        dp(i) = (d(i) - a(i)*dp(i-1)) / m
    end do

    ! solucionem a partir de x_n

    x(n) = dp(n)
    do i = n-1, 1, -1
        x(i) = dp(i) - cp(i)*x(i+1)
    end do

end subroutine thomas