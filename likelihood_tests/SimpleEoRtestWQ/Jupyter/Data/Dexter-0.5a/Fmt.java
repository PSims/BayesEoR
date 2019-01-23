// Fmt - some simple single-arg sprintf-like routines
//
// Copyright (C) 1996 by Jef Poskanzer <jef@acme.com>.  All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.
//
// Visit the ACME Labs Java page for up-to-date versions of this and other
// fine Java utilities: http://www.acme.com/java/

class Fmt
{
  // Flags.
  /// Zero-fill.
  public static final int ZF = 1;
  /// Left justify.
  public static final int LJ = 2;
  /// Hexadecimal.
  public static final int HX = 4;
  /// Octal.
  public static final int OC = 8;
  // Was a number - internal use.
  private static final int WN = 16;

  public static String fmt( double d )
	{
	return fmt( d, 0, 0, 0 );
	}
  public static String fmt( double d, int minWidth )
	{
	return fmt( d, minWidth, 0, 0 );
	}
  public static String fmt( double d, int minWidth, int sigFigs )
	{
	return fmt( d, minWidth, sigFigs, 0 );
	}
  public static String fmt( double d, int minWidth, int sigFigs, int flags )
	{
	if ( sigFigs != 0 )
	    return fmt(
		sigFigFix( Double.toString( d ), sigFigs ), minWidth,
		flags|WN );
	else
	    return fmt( Double.toString( d ), minWidth, flags|WN );
	}

  public static String fmt( String s, int minWidth, int flags )
	{
	int len = s.length();
	boolean zeroFill = ( ( flags & ZF ) != 0 );
	boolean leftJustify = ( ( flags & LJ ) != 0 );
	boolean hexadecimal = ( ( flags & HX ) != 0 );
	boolean octal = ( ( flags & OC ) != 0 );
	boolean wasNumber = ( ( flags & WN ) != 0 );
	if ( ( hexadecimal || octal || zeroFill ) && ! wasNumber )
	    throw new InternalError( "Acme.Fmt: number flag on a non-number" );
	if ( zeroFill && leftJustify )
	    throw new InternalError( "Acme.Fmt: zero-fill left-justify is silly" );
	if ( hexadecimal && octal )
	    throw new InternalError( "Acme.Fmt: can't do both hex and octal" );
	if ( len >= minWidth )
	    return s;
	int fillWidth = minWidth - len;
	StringBuffer fill = new StringBuffer( fillWidth );
	for ( int i = 0; i < fillWidth; ++i )
	    if ( zeroFill )
		fill.append( '0' );
	    else
		fill.append( ' ' );
	if ( leftJustify )
	    return s + fill;
	else if ( zeroFill && s.startsWith( "-" ) )
	    return "-" + fill + s.substring( 1 );
	else
	    return fill + s;
	}

  private static String sigFigFix( String s, int sigFigs )
	{
	// First dissect the floating-point number string into sign,
	// integer part, fraction part, and exponent.
	String sign;
	String unsigned;
	if ( s.startsWith( "-" ) || s.startsWith( "+" ) )
	    {
	    sign = s.substring( 0, 1 );
	    unsigned = s.substring( 1 );
	    }
	else
	    {
	    sign = "";
	    unsigned = s;
	    }
	String mantissa;
	String exponent;
	int eInd = unsigned.indexOf( 'e' );
	if ( eInd == -1 )	// it may be 'e' or 'E'
	    eInd = unsigned.indexOf( 'E' );
	if ( eInd == -1 )
	    {
	    mantissa = unsigned;
	    exponent = "";
	    }
	else
	    {
	    mantissa = unsigned.substring( 0, eInd );
	    exponent = unsigned.substring( eInd );
	    }
	StringBuffer number, fraction;
	int dotInd = mantissa.indexOf( '.' );
	if ( dotInd == -1 )
	    {
	    number = new StringBuffer( mantissa );
	    fraction = new StringBuffer( "" );
	    }
	else
	    {
	    number = new StringBuffer( mantissa.substring( 0, dotInd ) );
	    fraction = new StringBuffer( mantissa.substring( dotInd + 1 ) );
	    }

	int numFigs = number.length();
	int fracFigs = fraction.length();
	if ( ( numFigs == 0 || Integer.parseInt(number.toString())==0 ) && fracFigs > 0 )
	    {
	    // Don't count leading zeros in the fraction.
	    numFigs = 0;
	    for ( int i = 0; i < fraction.length(); ++i )
		{
		if ( fraction.charAt( i ) != '0' )
		    break;
		--fracFigs;
		}
	    }
	int mantFigs = numFigs + fracFigs;
	if ( sigFigs > mantFigs )
	    {
	    // We want more figures; just append zeros to the fraction.
	    for ( int i = mantFigs; i < sigFigs; ++i )
		fraction.append( '0' );
	    }
	else if ( sigFigs < mantFigs && sigFigs >= numFigs )
	    {
	    // Want fewer figures in the fraction; chop.
	    fraction.setLength(
		fraction.length() - ( fracFigs - ( sigFigs - numFigs ) ) );
	    // Round?
	    }
	else if ( sigFigs < numFigs )
	    {
	    // Want fewer figures in the number; turn them to zeros.
	    fraction.setLength( 0 );	// should already be zero, but make sure
	    for ( int i = sigFigs; i < numFigs; ++i )
		number.setCharAt( i, '0' );
	    // Round?
	    }
	// Else sigFigs == mantFigs, which is fine.

	if ( fraction.length() == 0 )
	    return sign + number + exponent;
	else
	    return sign + number + "." + fraction + exponent;
	}


  /// Improved version of Double.toString(), returns more decimal places.
  // <P>
  // The JDK 1.0.2 version of Double.toString() returns only six decimal
  // places on some systems.  In JDK 1.1 full precision is returned on
  // all platforms.
  // @deprecated
  // @see java.lang.Double#toString
  public static String doubleToString( double d )
	{
	// Handle special numbers first, to avoid complications.
	if ( Double.isNaN( d ) )
	    return "NaN";
	if ( d == Double.NEGATIVE_INFINITY )
	    return "-Inf";
	if ( d == Double.POSITIVE_INFINITY )
	    return "Inf";

	// Grab the sign, and then make the number positive for simplicity.
	boolean negative = false;
	if ( d < 0.0D )
	    {
	    negative = true;
	    d = -d;
	    }

	// Get the native version of the unsigned value, as a template.
	String unsStr = Double.toString( d );

	// Dissect out the exponent.
	String mantStr, expStr;
	int exp;
	int eInd = unsStr.indexOf( 'e' );
	if ( eInd == -1 )	// it may be 'e' or 'E'
	    eInd = unsStr.indexOf( 'E' );
	if ( eInd == -1 )
	    {
	    mantStr = unsStr;
	    expStr = "";
	    exp = 0;
	    }
	else
	    {
	    mantStr = unsStr.substring( 0, eInd );
	    expStr = unsStr.substring( eInd + 1 );
	    if ( expStr.startsWith( "+" ) )
		exp = Integer.parseInt( expStr.substring( 1 ) );
	    else
		exp = Integer.parseInt( expStr );
	    }

	// Dissect out the number part.
	String numStr;
	int dotInd = mantStr.indexOf( '.' );
	if ( dotInd == -1 )
	    numStr = mantStr;
	else
	    numStr = mantStr.substring( 0, dotInd );
	long num;
	if ( numStr.length() == 0 )
	    num = 0;
	else
	    num = Integer.parseInt( numStr );

	// Build the new mantissa.
	StringBuffer newMantBuf = new StringBuffer( numStr + "." );
	double p = Math.pow( 10, exp );
	double frac = d - num * p;
	String digits = "0123456789";
	int nDigits = 16 - numStr.length();	// about 16 digits in a double
	for ( int i = 0; i < nDigits; ++i )
	    {
	    p /= 10.0D;
	    int dig = (int) ( frac / p );
	    if ( dig < 0 ) dig = 0;
	    if ( dig > 9 ) dig = 9;
	    newMantBuf.append( digits.charAt( dig ) );
	    frac -= dig * p;
	    }

	if ( (int) ( frac / p + 0.5D ) == 1 )
	    {
	    // Round up.
	    boolean roundMore = true;
	    for ( int i = newMantBuf.length() - 1; i >= 0; --i )
		{
		int dig = digits.indexOf( newMantBuf.charAt( i ) );
		if ( dig == -1 )
		    continue;
		++dig;
		if ( dig == 10 )
		    {
		    newMantBuf.setCharAt( i, '0' );
		    continue;
		    }
		newMantBuf.setCharAt( i, digits.charAt( dig ) );
		roundMore = false;
		break;
		}
	    if ( roundMore )
		{
		// If this happens, we need to prepend a 1.  But I haven't
		// found a test case yet, so I'm leaving it out for now.
		// But if you get this message, please let me know!
		newMantBuf.append( "ROUNDMORE" );
		}
	    }

	// Chop any trailing zeros.
	int len = newMantBuf.length();
	while ( newMantBuf.charAt( len - 1 ) == '0' )
	    newMantBuf.setLength( --len );
	// And chop a trailing dot, if any.
	if ( newMantBuf.charAt( len - 1 ) == '.' )
	    newMantBuf.setLength( --len );

	// Done.
	return ( negative ? "-" : "" ) +
	       newMantBuf +
	       ( expStr.length() != 0 ? ( "e" + expStr ) : "" );
	}
}

