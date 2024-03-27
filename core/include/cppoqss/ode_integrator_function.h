#ifndef ODE_INTEGRATOR_FUNCTION_H 
#define ODE_INTEGRATOR_FUNCTION_H 


#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/util/detail/less_with_sign.hpp>


namespace cppoqss { namespace ode_integrator_function {


/**
 * Customized for process inside steps and error calculation
 */


using namespace boost::numeric;
using namespace boost::numeric::odeint;
using namespace boost::numeric::odeint::detail;


/*
 * error stepper category dispatcher
 */
template<
class ErrorStepper ,
class ErrorChecker = default_error_checker< typename ErrorStepper::value_type ,
    typename ErrorStepper::algebra_type ,
    typename ErrorStepper::operations_type > ,
class StepAdjuster = default_step_adjuster< typename ErrorStepper::value_type ,
    typename ErrorStepper::time_type > ,
class Resizer = typename ErrorStepper::resizer_type ,
class ErrorStepperCategory = typename ErrorStepper::stepper_category
>
class controlled_runge_kutta_spmat ;


/*
 * explicit stepper version
 *
 * this class introduces the following try_step overloads
    * try_step( sys , x , t , dt )
    * try_step( sys , x , dxdt , t , dt )
    * try_step( sys , in , t , out , dt )
    * try_step( sys , in , dxdt , t , out , dt )
 */
/**
 * \brief Implements step size control for Runge-Kutta steppers with error 
 * estimation.
 *
 * This class implements the step size control for standard Runge-Kutta 
 * steppers with error estimation.
 *
 * \tparam ErrorStepper The stepper type with error estimation, has to fulfill the ErrorStepper concept.
 * \tparam ErrorChecker The error checker
 * \tparam Resizer The resizer policy type.
 */
template<
class ErrorStepper,
class ErrorChecker,
class StepAdjuster,
class Resizer
>
class controlled_runge_kutta_spmat< ErrorStepper , ErrorChecker , StepAdjuster, Resizer ,
        explicit_error_stepper_tag >
{

public:

    typedef ErrorStepper stepper_type;
    typedef typename stepper_type::state_type state_type;
    typedef typename stepper_type::value_type value_type;
    typedef typename stepper_type::deriv_type deriv_type;
    typedef typename stepper_type::time_type time_type;
    typedef typename stepper_type::algebra_type algebra_type;
    typedef typename stepper_type::operations_type operations_type;
    typedef Resizer resizer_type;
    typedef ErrorChecker error_checker_type;
    typedef StepAdjuster step_adjuster_type;
    typedef explicit_controlled_stepper_tag stepper_category;

#ifndef DOXYGEN_SKIP
    typedef typename stepper_type::wrapped_state_type wrapped_state_type;
    typedef typename stepper_type::wrapped_deriv_type wrapped_deriv_type;

    typedef controlled_runge_kutta< ErrorStepper , ErrorChecker , StepAdjuster ,
            Resizer , explicit_error_stepper_tag > controlled_stepper_type;
#endif //DOXYGEN_SKIP


    /**
     * \brief Constructs the controlled Runge-Kutta stepper.
     * \param error_checker An instance of the error checker.
     * \param stepper An instance of the underlying stepper.
     */
    controlled_runge_kutta_spmat(
            const error_checker_type &error_checker = error_checker_type( ) ,
            const step_adjuster_type &step_adjuster = step_adjuster_type() ,
            const stepper_type &stepper = stepper_type( )
    )
        : m_stepper(stepper), m_error_checker(error_checker) , m_step_adjuster(step_adjuster)
    { }



    /*
     * Version 1 : try_step( sys , x , t , dt )
     *
     * The overloads are needed to solve the forwarding problem
     */
    /**
     * \brief Tries to perform one step.
     *
     * This method tries to do one step with step size dt. If the error estimate
     * is to large, the step is rejected and the method returns fail and the 
     * step size dt is reduced. If the error estimate is acceptably small, the
     * step is performed, success is returned and dt might be increased to make 
     * the steps as large as possible. This method also updates t if a step is
     * performed.
     *
     * \param system The system function to solve, hence the r.h.s. of the ODE. It must fulfill the
     *               Simple System concept.
     * \param x The state of the ODE which should be solved. Overwritten if 
     * the step is successful.
     * \param t The value of the time. Updated if the step is successful.
     * \param dt The step size. Updated.
     * \return success if the step was accepted, fail otherwise.
     */
    template< class System , class StateInOut >
    controlled_step_result try_step( System system , StateInOut &x , time_type &t , time_type &dt )
    {
        return try_step_v1( system , x , t, dt );
    }

    /**
     * \brief Tries to perform one step. Solves the forwarding problem and 
     * allows for using boost range as state_type.
     *
     * This method tries to do one step with step size dt. If the error estimate
     * is to large, the step is rejected and the method returns fail and the 
     * step size dt is reduced. If the error estimate is acceptably small, the
     * step is performed, success is returned and dt might be increased to make 
     * the steps as large as possible. This method also updates t if a step is
     * performed.
     *
     * \param system The system function to solve, hence the r.h.s. of the ODE. It must fulfill the
     *               Simple System concept.
     * \param x The state of the ODE which should be solved. Overwritten if 
     * the step is successful. Can be a boost range.
     * \param t The value of the time. Updated if the step is successful.
     * \param dt The step size. Updated.
     * \return success if the step was accepted, fail otherwise.
     */
    template< class System , class StateInOut >
    controlled_step_result try_step( System system , const StateInOut &x , time_type &t , time_type &dt )
    {
        return try_step_v1( system , x , t, dt );
    }



    /*
     * Version 2 : try_step( sys , x , dxdt , t , dt )
     *
     * this version does not solve the forwarding problem, boost.range can not be used
     */
    /**
     * \brief Tries to perform one step.
     *
     * This method tries to do one step with step size dt. If the error estimate
     * is to large, the step is rejected and the method returns fail and the 
     * step size dt is reduced. If the error estimate is acceptably small, the
     * step is performed, success is returned and dt might be increased to make 
     * the steps as large as possible. This method also updates t if a step is
     * performed.
     *
     * \param system The system function to solve, hence the r.h.s. of the ODE. It must fulfill the
     *               Simple System concept.
     * \param x The state of the ODE which should be solved. Overwritten if 
     * the step is successful.
     * \param dxdt The derivative of state.
     * \param t The value of the time. Updated if the step is successful.
     * \param dt The step size. Updated.
     * \return success if the step was accepted, fail otherwise.
     */
    template< class System , class StateInOut , class DerivIn >
    controlled_step_result try_step( System system , StateInOut &x , const DerivIn &dxdt , time_type &t , time_type &dt )
    {
        m_xnew_resizer.adjust_size( x , detail::bind( &controlled_runge_kutta_spmat::template resize_m_xnew_impl< StateInOut > , detail::ref( *this ) , detail::_1 ) );
        controlled_step_result res = try_step( system , x , dxdt , t , m_xnew.m_v , dt );
        if( res == success )
        {
            boost::numeric::odeint::copy( m_xnew.m_v , x );
        }
        return res;
    }

    /*
     * Version 3 : try_step( sys , in , t , out , dt )
     *
     * this version does not solve the forwarding problem, boost.range can not be used
     *
     * the disable is needed to avoid ambiguous overloads if state_type = time_type
     */
    /**
     * \brief Tries to perform one step.
     *
     * \note This method is disabled if state_type=time_type to avoid ambiguity.
     *
     * This method tries to do one step with step size dt. If the error estimate
     * is to large, the step is rejected and the method returns fail and the 
     * step size dt is reduced. If the error estimate is acceptably small, the
     * step is performed, success is returned and dt might be increased to make 
     * the steps as large as possible. This method also updates t if a step is
     * performed.
     *
     * \param system The system function to solve, hence the r.h.s. of the ODE. It must fulfill the
     *               Simple System concept.
     * \param in The state of the ODE which should be solved.
     * \param t The value of the time. Updated if the step is successful.
     * \param out Used to store the result of the step.
     * \param dt The step size. Updated.
     * \return success if the step was accepted, fail otherwise.
     */
    template< class System , class StateIn , class StateOut >
    typename boost::disable_if< boost::is_same< StateIn , time_type > , controlled_step_result >::type
    try_step( System system , const StateIn &in , time_type &t , StateOut &out , time_type &dt )
    {
        typename odeint::unwrap_reference< System >::type &sys = system;
        m_dxdt_resizer.adjust_size( in , detail::bind( &controlled_runge_kutta_spmat::template resize_m_dxdt_impl< StateIn > , detail::ref( *this ) , detail::_1 ) );
        sys( in , m_dxdt.m_v , t );
        return try_step( system , in , m_dxdt.m_v , t , out , dt );
    }


    /*
     * Version 4 : try_step( sys , in , dxdt , t , out , dt )
     *
     * this version does not solve the forwarding problem, boost.range can not be used
     */
    /**
     * \brief Tries to perform one step.
     *
     * This method tries to do one step with step size dt. If the error estimate
     * is to large, the step is rejected and the method returns fail and the 
     * step size dt is reduced. If the error estimate is acceptably small, the
     * step is performed, success is returned and dt might be increased to make 
     * the steps as large as possible. This method also updates t if a step is
     * performed.
     *
     * \param system The system function to solve, hence the r.h.s. of the ODE. It must fulfill the
     *               Simple System concept.
     * \param in The state of the ODE which should be solved.
     * \param dxdt The derivative of state.
     * \param t The value of the time. Updated if the step is successful.
     * \param out Used to store the result of the step.
     * \param dt The step size. Updated.
     * \return success if the step was accepted, fail otherwise.
     */
    template< class System , class StateIn , class DerivIn , class StateOut >
    controlled_step_result try_step( System system , const StateIn &in , const DerivIn &dxdt , time_type &t , StateOut &out , time_type &dt )
    {
        unwrapped_step_adjuster &step_adjuster = m_step_adjuster;
        if( !step_adjuster.check_step_size_limit(dt) )
        {
            // given dt was above step size limit - adjust and return fail;
            dt = step_adjuster.get_max_dt();
            return fail;
        }

        m_xerr_resizer.adjust_size( in , detail::bind( &controlled_runge_kutta_spmat::template resize_m_xerr_impl< StateIn > , detail::ref( *this ) , detail::_1 ) );

        // do one step with error calculation
        m_stepper.do_step( system , in , dxdt , t , out , dt , m_xerr.m_v );

        // modified
        value_type max_rel_err = m_error_checker.error( m_stepper.algebra() , in , (in*(-1.0) + out)*(1.0/dt) , m_xerr.m_v , dt );

        if( max_rel_err > 1.0 )
        {
            // error too big, decrease step size and reject this step
            dt = step_adjuster.decrease_step(dt, max_rel_err, m_stepper.error_order());
            return fail;
        } else
        {
            // otherwise, increase step size and accept
            t += dt;
            dt = step_adjuster.increase_step(dt, max_rel_err, m_stepper.stepper_order());
            return success;
        }
    }

    /**
     * \brief Adjust the size of all temporaries in the stepper manually.
     * \param x A state from which the size of the temporaries to be resized is deduced.
     */
    template< class StateType >
    void adjust_size( const StateType &x )
    {
        resize_m_xerr_impl( x );
        resize_m_dxdt_impl( x );
        resize_m_xnew_impl( x );
        m_stepper.adjust_size( x );
    }

    /**
     * \brief Returns the instance of the underlying stepper.
     * \returns The instance of the underlying stepper.
     */
    stepper_type& stepper( void )
    {
        return m_stepper;
    }

    /**
     * \brief Returns the instance of the underlying stepper.
     * \returns The instance of the underlying stepper.
     */
    const stepper_type& stepper( void ) const
    {
        return m_stepper;
    }

private:


    template< class System , class StateInOut >
    controlled_step_result try_step_v1( System system , StateInOut &x , time_type &t , time_type &dt )
    {
        typename odeint::unwrap_reference< System >::type &sys = system;
        m_dxdt_resizer.adjust_size( x , detail::bind( &controlled_runge_kutta_spmat::template resize_m_dxdt_impl< StateInOut > , detail::ref( *this ) , detail::_1 ) );
        sys( x , m_dxdt.m_v ,t );
        return try_step( system , x , m_dxdt.m_v , t , dt );
    }

    template< class StateIn >
    bool resize_m_xerr_impl( const StateIn &x )
    {
        return adjust_size_by_resizeability( m_xerr , x , typename is_resizeable<state_type>::type() );
    }

    template< class StateIn >
    bool resize_m_dxdt_impl( const StateIn &x )
    {
        return adjust_size_by_resizeability( m_dxdt , x , typename is_resizeable<deriv_type>::type() );
    }

    template< class StateIn >
    bool resize_m_xnew_impl( const StateIn &x )
    {
        return adjust_size_by_resizeability( m_xnew , x , typename is_resizeable<state_type>::type() );
    }



    stepper_type m_stepper;
    error_checker_type m_error_checker;
    step_adjuster_type m_step_adjuster;
    typedef typename unwrap_reference< step_adjuster_type >::type unwrapped_step_adjuster;

    resizer_type m_dxdt_resizer;
    resizer_type m_xerr_resizer;
    resizer_type m_xnew_resizer;

    wrapped_deriv_type m_dxdt;
    wrapped_state_type m_xerr;
    wrapped_state_type m_xnew;
};


template< class Stepper >
controlled_runge_kutta_spmat< Stepper > make_controlled_rk_spmat(
    typename Stepper::value_type abs_error ,
    typename Stepper::value_type rel_error ,
    const Stepper & stepper = Stepper() )
{
  typedef Stepper stepper_type;
  typedef controlled_runge_kutta_spmat< stepper_type > controller_type;
  typedef controller_factory< stepper_type , controller_type > factory_type;
  factory_type factory;
  return factory( abs_error , rel_error , stepper );
}

/*
 * integrate_times for simple stepper
 */
template<class Stepper, class System, class State, class TimeIterator, class Time, class Observer>
size_t integrate_times_mod(
    Stepper stepper , System system , State &start_state ,
    TimeIterator start_time , TimeIterator end_time , Time dt ,
    Observer observer , stepper_tag
    )
{
    typedef typename odeint::unwrap_reference< Stepper >::type stepper_type;
    typedef typename odeint::unwrap_reference< Observer >::type observer_type;

    stepper_type &st = stepper;
    observer_type &obs = observer;
    typedef typename unit_value_type<Time>::type time_type;

    size_t steps = 0;
    Time current_dt = dt;
    while( true )
    {
	Time current_time = *start_time++;
	obs( start_state , current_time );
	if( start_time == end_time )
	    break;
	while( less_with_sign( current_time , static_cast<time_type>(*start_time) , current_dt ) )
	{
	    current_dt = min_abs( dt , *start_time - current_time );
	    obs.record_time_step(current_time, current_dt);
	    st.do_step( system , start_state , current_time , current_dt );
	    current_time += current_dt;
	    steps++;
	}
    }
    return steps;
}

/*
 * integrate_times for controlled stepper
 */
template< class Stepper , class System , class State , class TimeIterator , class Time , class Observer >
size_t integrate_times_mod(
    Stepper stepper , System system , State &start_state ,
    TimeIterator start_time , TimeIterator end_time , Time dt ,
    Observer observer , controlled_stepper_tag
    )
{
  typename odeint::unwrap_reference< Observer >::type &obs = observer;
  typename odeint::unwrap_reference< Stepper >::type &st = stepper;
  typedef typename unit_value_type<Time>::type time_type;

  failed_step_checker fail_checker;  // to throw a runtime_error if step size adjustment fails
  size_t steps = 0;
  while( true )
  {
    Time current_time = *start_time++;
    obs( start_state , current_time );
    if( start_time == end_time )
      break;
    while( less_with_sign( current_time , static_cast<time_type>(*start_time) , dt ) )
    {
      Time buffer_current_time = current_time;

      // adjust stepsize to end up exactly at the observation point
      Time current_dt = min_abs( dt , *start_time - current_time );
      Time dt_tried = current_dt;
      if( st.try_step( system , start_state , current_time , current_dt ) == success )
      {
        ++steps;
        // successful step -> reset the fail counter, see #173
        fail_checker.reset();
        // continue with the original step size if dt was reduced due to observation
        dt = max_abs( dt , current_dt );

        Time updated_time = current_time;
        Time actual_dt = updated_time - buffer_current_time;
        obs.record_time_step(buffer_current_time, actual_dt);
      }
      else
      {
        fail_checker();  // check for possible overflow of failed steps in step size adjustment
        dt = current_dt;
        obs.record_failed_time_step(current_time, dt_tried, dt);
      }
    }
  }
  return steps;
}

template< class Stepper , class System , class State , class TimeIterator , class Time , class Observer >
size_t integrate_times(
    Stepper stepper , System system , State &start_state ,
    TimeIterator times_start , TimeIterator times_end , Time dt ,
    Observer observer )
{
    typedef typename odeint::unwrap_reference< Stepper >::type::stepper_category stepper_category;
    // simply don't use checked_* adapters
    return integrate_times_mod(
	    stepper , system , start_state ,
	    times_start , times_end , dt ,
	    observer , stepper_category() );
}

} // ode_integrator_function

} // namespace cppoqss


namespace boost { namespace numeric { namespace odeint {


// controller factory for controlled_runge_kutta_spmat
template< class Stepper >
struct controller_factory< Stepper , cppoqss::ode_integrator_function::controlled_runge_kutta_spmat< Stepper > >
{
    typedef Stepper stepper_type;
    typedef cppoqss::ode_integrator_function::controlled_runge_kutta_spmat< stepper_type > controller_type;
    typedef typename controller_type::error_checker_type error_checker_type;
    typedef typename controller_type::step_adjuster_type step_adjuster_type;
    typedef typename stepper_type::value_type value_type;
    typedef typename stepper_type::value_type time_type;

    controller_type operator()( value_type abs_error , value_type rel_error , const stepper_type &stepper )
    {
        return controller_type( error_checker_type( abs_error , rel_error ) ,
                                step_adjuster_type() , stepper );
    }

    controller_type operator()( value_type abs_error , value_type rel_error ,
                                time_type max_dt, const stepper_type &stepper )
    {
        return controller_type( error_checker_type( abs_error , rel_error ) ,
                                step_adjuster_type(max_dt) , stepper );
    }
};


}}} // namespace boost::numeric::odeint


#endif
