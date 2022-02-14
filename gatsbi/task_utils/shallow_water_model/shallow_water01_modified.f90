!***************************************************************************
!*                                                                         *
!*             1-D Shallow-Water equations semi-implicit solver            *
!*                                                                         *
!*                           Kai Logemann, HZG, October 2020               *
!*                                                                         *
!***************************************************************************
subroutine shallow_water(d, outdir)
  integer i,nx,ip,im,outdir
  parameter(nx=100)

  real dt,dx,u(nx),un(nx),zeta(nx),zetan(nx),h(nx),cd,g,uint(nx),hm,us(nx),uints(nx),div(nx)
  real time,rhour,umin,umax,zmin,zmax,wimp,ce(nx),cw(nx),zetnew,zs(nx),dif,maxdif,d(nx)
  integer hour,it,ho


  ! set constants ==================================

  dt = 300. ! time step [s]
  dx = 10.e3 ! grid spacing [m]

  ! set boundary for water depth profile
  d(1) = -10. ! dry boundary cell
  d(nx) = -10. ! dry boundary cell

  h = d ! thickness of water column [m]

  cd = 1.e-3 ! bottom drag coefficient
  g = 9.81 ! acceleration due to gravity [m/s**2]

  uint = 0. ! vertically integrated velocity [m**2/s]

  wimp = 0.5 ! weight between time levels


  ! initalisation ====================================

  u = 0. ! velocity [m/s]
  un = 0. ! velocity at new time step [m/s]
  us = 0. ! interim solution of velocity [m/s]

  zeta = 0. ! sea surface elevation [m]
  zetan = 0. ! sea surface elevation at new time level [m]

  zeta(2) = 0.1 ! initial disturbance
  h(2) = d(2) + zeta(2)

  time = 0. ! run time [s]

  ! ======================================================

  ho = 0
  call output(ho) ! write output to directory "dat"

  ! ======================================================

  ! begin of time loop

  100 time = time + dt
  rhour = time/3600.
  hour = int(rhour)

  ! equation of motion => us (interim solution)
  do i = 1,nx,1
    ip = min(nx,i+1)
    if((d(i).gt.0.1).and.(d(ip).gt.0.1)) then
      hm = 0.5*(h(i)+h(ip))
      us(i) = u(i)- dt*cd/hm*u(i)*abs(u(i)) - (1.-wimp)*dt*g*(zeta(ip)-zeta(i))/dx 
    else
      us(i) = 0.
    endif
  enddo

  ! vertical integrals of velocity => uint, and interim solution uints

  do i = 1,nx,1
    ip = min(nx,i+1)
    if((d(i).gt.0.1).and.(d(ip).gt.0.1)) then
      hm = 0.5*(h(i)+h(ip))
      uint(i) = hm*u(i)
    uints(i) = hm*us(i)
    else
      uint(i) = 0.
    uints(i) = 0.
    endif
  enddo

  ! equation of continuity, first compute div

  div = 0.
  do i = 1,nx,1
    if(d(i).gt.0.1) then
      im = max(1,i-1)
      div(i) = -dt*(1.-wimp)*(uint(i)-uint(im))/dx  - dt*wimp*(uints(i)-uints(im))/dx
    endif
  enddo

  ! now compute coefficients for zetan system of equations: ce and cw

  ce = 0.
  cw = 0.

  do i = 1,nx,1
    im = max(1,i-1)
    ip = min(nx,i+1)
    
    if((d(i).gt.0.1).and.(d(ip).gt.0.1)) then
      ce(i) = dt*dt*wimp*wimp*g*0.5*(h(i)+h(ip))/(dx*dx)
    endif
    
    if((d(i).gt.0.1).and.(d(im).gt.0.1)) then
      cw(i) = dt*dt*wimp*wimp*g*0.5*(h(i)+h(im))/(dx*dx)
    endif
    
  enddo

  ! iterative solution of the system of eqations

  it = 0

  zs = zeta

  1 continue
  maxdif = 0.
  it = it+1

  do i = 1,nx,1
    im = max(1,i-1)
    ip = min(nx,i+1)
    if(d(i).gt.0.1) then
      zetnew = 1./(1. + ce(i) + cw(i))*(zeta(i) + div(i) + ce(i)*zetan(ip) + cw(i)*zetan(im))
    
    dif = abs(zetnew-zs(i))
    maxdif = max(maxdif,dif)
    zs(i) = zetnew
    else
      zs(i) = 0.
    endif
  enddo

  zetan = zs ! update iterative solution (could be also done in the loop above, but would lead to problems when parallelisation is applied.)

  !write(*,*) 'it,maxdif',it,maxdif
  !read(*,*)

  if(maxdif.gt.1.e-6) goto 1 ! convergence check

  !write(*,*) 'iterations: ',it

  ! ===============================================================
  ! finally compute compute velocity at new time step un

  do i = 1,nx,1
    ip = min(nx,i+1)
    if((d(i).gt.0.1).and.(d(ip).gt.0.1)) then
      un(i) = us(i) - wimp*dt*g*(zetan(ip)-zetan(i))/dx 
    else
      un(i) = 0.
    endif
  enddo

  ! ================================================================
  ! swap time level

  u = un
  zeta = zetan

  do i = 1,nx,1
    if(d(i).gt.0.1) then
      h(i) = d(i) + zeta(i)
    endif
  enddo
   
  ! ================================================
  ! compute min,max for screen output

  umin = 1.e6
  umax = -1.e6
  zmin = 1.e6
  zmax = -1.e6

  do i = 1,nx,1
    if(d(i).gt.0.1) then
      umin = min(umin,u(i))
    umax = max(umax,u(i))
    zmin = min(zmin,zeta(i))
    zmax = max(zmax,zeta(i))
    endif
  enddo

  !write(*,*) 'rhour,umin,umax,zmin,zmax',rhour,umin,umax,zmin,zmax


  ! =========================================================
  ! write hourly output

  if(hour.ne.ho) then
    ho = hour
    call output(ho)
  endif

  ! ================================================

  if(hour.lt.100) goto 100 ! continuation criterium of time loop

  !stop

  contains

  ! ========================================================================

    subroutine output(ho)
    integer ho
    character*50 ufile,zfile
    
    ufile = 'DDDDDDD/uHHH.dat'
    zfile = 'DDDDDDD/zHHH.dat'
    write(ufile(1:7),'(i7.7)') outdir
    write(zfile(1:7),'(i7.7)') outdir
    write(ufile(10:12),'(i3.3)') ho
    write(zfile(10:12),'(i3.3)') ho
    
    open(1, file = ufile)
    do i = 1,nx,1
      write(1,'(f12.4)') u(i)
    enddo
    close(1)
    
    open(1, file = zfile)
    do i = 1,nx,1
      write(1,'(f12.4)') zeta(i)
    enddo
    close(1)

    end subroutine output
end subroutine shallow_water 
