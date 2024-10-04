import { NextRequest, NextResponse } from 'next/server';
import { del } from '@vercel/blob';
import { z } from 'zod';

const paramsSchema = z.object({
  space_x: z.number(),
  space_y: z.number(),
  space_z: z.number(),
  a_min: z.number(),
  a_max: z.number(),
  b_min: z.number(),
  b_max: z.number(),
  roi_x: z.number(),
  roi_y: z.number(),
  roi_z: z.number(),
  num_samples: z.number(),
});

// Set the max time of the request
export const maxDuration = 300;

export async function POST(request: NextRequest) {
  let url;

  try {
    const formData = await request.formData();
    const file = formData.get('file') as string;
    const selectedTargetsRaw = formData.get('selectedTargets') as string;
    const paramsRaw = formData.get('params') as string;

    console.log('formData:', formData);
    console.log('file:', file);
    console.log('paramsRaw:', paramsRaw);
    console.log('selectedTargetsRaw:', selectedTargetsRaw);

    // JSON 데이터를 파싱
    const params = JSON.parse(paramsRaw);
    const selectedTargets = JSON.parse(selectedTargetsRaw);

    url = file;

    // Validate inputs
    if (!file || !selectedTargets) {
      return NextResponse.json(
        { error: 'Missing required fields' },
        { status: 400 }
      );
    }

    // Validate parameters using safeParse
    const parsedParams = paramsSchema.safeParse(params);
    if (!parsedParams.success) {
      return NextResponse.json(
        { error: 'Invalid parameters: ' + parsedParams.error.message },
        { status: 400 }
      );
    }

    const res = await fetch(`${process.env.RUNPOD_ENDPOINT}/runsync`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${process.env.RUNPOD_ENDPOINT_KEY}`,
      },
      body: JSON.stringify({
        input: {
          ...params,
          url: file,
          targets: JSON.parse(selectedTargets),
        },
      }),
    }).then((r) => r.json());

    if (res.error) {
      throw new Error(
        'An error occurred while processing the image: ' + res.error
      );
    }

    // Return the download URL to the client
    return NextResponse.json(res.output);
  } catch (error) {
    console.error('Error processing CT image:', error);
    return NextResponse.json(
      { error: 'An error occurred while processing the image' },
      { status: 500 }
    );
  } finally {
    if (url) {
      await del(url);
    }
  }
}
